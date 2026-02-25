#include "common/tester.h"
#include "common/util.h"
#include <cuda_runtime.h>


using namespace nvcuda;
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])

template <const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C, int M, int N, int K) {

    /*256 threads(8 warps) per block.*/
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M;  // 16 x 4 = 64
    constexpr int BN = WMMA_N * WMMA_TILE_N;  // 16 x 2 = 32
    constexpr int BK = WMMA_K;  // 16
    __shared__ half tileA[BM][BK], tileB[WMMA_K][BN];  // 64 x 16 = 1KB, 16 x 32 x 2 = 1KB

    // 要保证相同的warp下thread执行相同的指令
    // warp_id 0 -> warp_m 0, warp_n 0
    // warp_id 1 -> warp_m 0, warp_n 1
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // 256线程分别load tileA=64x16, tileB=16x32
    // 64 x 16 / 256 = 4 half4, 16x32/256=2, half2
    // tileA, 每个线程load 4 half, 每行需要4线程, 64行, 共256线程
    const int load_smem_a_m = tid / 4;
    const int load_smem_a_k = (tid % 4) * 4;
    // tileB, 每个线程load 2 half, 每行需要8线程, 32行, 共256线程
    const int load_smem_b_k = tid / 16;
    const int load_smem_b_n = (tid % 16) * 2;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;

    if(load_gmem_a_m >= M && load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

#pragma unroll
    for(int k = 0; k < NUM_K_TILES; ++k) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;  // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;  // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // 64 bits sync memory issues gmem_a -> smem_a
        LDST64BITS(tileA[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_addr]);
        LDST32BITS(tileB[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_addr]);
        __syncthreads();

        wmma::load_matrix_sync(A_frag, &tileA[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(B_frag, &tileB[0][warp_n * WMMA_N], BN);
        
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }
    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N, wmma::mem_row_major);
    return;
}

void hgemm_wmma_m16n16k16_mma4x2(half *A, half *B, half *C, int M, int N, int K) {
    
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;

    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M,WMMA_N,WMMA_K,WMMA_TILE_M,WMMA_TILE_N,WARP_TILE_M,WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

int main() {

    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4");
    return 0;
}