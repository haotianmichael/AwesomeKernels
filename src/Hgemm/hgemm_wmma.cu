#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/util.h"

using namespace nvcuda;
#define LDST32BITS(pointer) (reinterpret_cast<half2*>(&(pointer))[0])
#define LDST64BITS(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
/*
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void store_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &a, fragment<...> &b, const fragment<...> &c, bool staf=false);
*/

/* naive wmma version*/
template<unsigned int WMMA_M = 16,
         unsigned int WMMA_N = 16,
         unsigned int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_kernel_v1(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    unsigned int row = blockIdx.y * WMMA_M;
    unsigned int col = blockIdx.x * WMMA_N;
    if(row >= M || col >= N) return;

    half *A_bck = A + row * K;
    half *B_bck = B + col;
    half *C_bck = C + row * N + col;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

#pragma unroll
    for(int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(A_frag, A_bck + k, K);
        wmma::load_matrix_sync(B_frag, B_bck + k * N, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    if(row + WMMA_M <= M && col + WMMA_N <= N) {
        wmma::store_matrix_sync(C_bck, C_frag, N, wmma::mem_row_major);
    }
}
void hgemm_wmma_m16n16k16_v1(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;
    dim3 block(32);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    hgemm_wmma_m16n16k16_kernel_v1<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(A, B, C, M, N, K); 
}

/* 
    v2: tiled sharedMem + mma4x2_warp1x1
    @block: 8个warp
    @mma4x2: 数据(64x16->4x2个[16,16]的tile)
    @warp1x1: wmma操作(一个tile用一个fragment处理->每个warp处理一个fragment)
*/
template<const int WMMA_M = 16,
         const int WMMA_N = 16,
         const int WMMA_K = 16,
         const int WMMA_TILE_M = 4,
         const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_kernel_v2(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    constexpr int BM = WMMA_M * WMMA_TILE_M; // 64
    constexpr int BN = WMMA_N * WMMA_TILE_N; // 32
    constexpr int BK = WMMA_K;  // 16
    __shared__ half tileA[BM][BK], tileB[BK][BN];

    unsigned int ty = threadIdx.y;
    unsigned int tx = threadIdx.x;
    const int tid = ty * blockDim.x + tx;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int NUM_K_TILES = (K + WMMA_K - 1) / WMMA_K;
  
    // 256threads 处理 tileA[64][16]  tileB[16][32];
    const int load_smem_a_m = tid / 4;
    const int load_smem_a_k = (tid % 4) * 4;
    const int load_smem_b_n = (tid % 16) * 2;
    const int load_smem_b_k = tid / 16;

    const int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    const int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    if(load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

#pragma unroll
    for(int k = 0; k < NUM_K_TILES; k++) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST64BITS(tileA[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_addr]);
        LDST32BITS(tileB[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_addr]);
        __syncthreads();
        wmma::load_matrix_sync(A_frag, &tileA[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(B_frag, &tileB[0][warp_n * WMMA_N], BN);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }

    const int store_gmem_a_m = blockIdx.y * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = blockIdx.x * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C+store_gmem_a_m*N + store_gmem_a_n, C_frag, N, wmma::mem_row_major);
    return;
}
void hgemm_wmma_m16n16k16_v2(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    dim3 block(256);
    dim3 grid((N + WMMA_N*WMMA_TILE_N - 1)/(WMMA_N*WMMA_TILE_N), (M + WMMA_M*WMMA_TILE_M - 1) / (WMMA_M*WMMA_TILE_M));
    hgemm_wmma_m16n16k16_kernel_v2<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

/*
    v3: tiled sharedMem + mma4x2_warp2x4
    @block: 8个warp
    @mma4x2: 数据(128x128->4x2个[32,64]的tile)
    @warp2x4: wmma操作(一个tile用2x4个fragment处理->每个warp处理8个fragment)
*/
#define LDST128BITS(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<const int WMMA_M = 16,
         const int WMMA_N = 16,
         const int WMMA_K = 16,
         const int WMMA_TILE_M = 4,
         const int WMMA_TILE_N = 2,
         const int WARP_TILE_M = 2,
         const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_kernel_v3(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned K) {

    const int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 128
    const int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    const int BK = WMMA_K;

    __shared__ half tileA[BM][BK];
    __shared__ half tileB[BK][BN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * blockDim.x + tx;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid % 2) * 8;
    const int load_smem_b_k = tid / 16;
    const int load_smem_b_n = (tid % 16) * 8;

    const int load_gmem_a_m = blockIdx.y * BM + load_smem_a_m;
    const int load_gmem_b_n = blockIdx.x * BN + load_smem_b_n;

    if(load_gmem_a_m >= M || load_gmem_b_n >= N) return;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_TILE_M][WARP_TILE_N];
    for(int i = 0; i < WARP_TILE_M; i ++) {
        for(int j = 0; j < WARP_TILE_N; j ++) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N,WMMA_K, half, wmma::row_major> A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N,WMMA_K, half, wmma::row_major> B_frag[WARP_TILE_N];

    const int K_NUM_TILES = (K + BK - 1) / BK;
    for(int s = 0; s < K_NUM_TILES; s ++) {
        int load_gmem_a_k = s * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = s * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
        LDST128BITS(tileA[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
        LDST128BITS(tileB[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
        __syncthreads();
        for(int i = 0; i < WARP_TILE_M; i++) {
            wmma::load_matrix_sync(A_frag[i], &tileA[warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M][0], BK);
        }
        for(int i = 0; i < WARP_TILE_N; i++) {
            wmma::load_matrix_sync(B_frag[i], &tileB[0][warp_n * WARP_TILE_N * WMMA_N + i * WMMA_N], BN);
        }

        for(int i = 0; i < WARP_TILE_M; i ++) {
            for(int j = 0; j < WARP_TILE_N; j ++) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < WARP_TILE_M; i ++) {
        for(int j = 0; j < WARP_TILE_N; j ++) {
            const int store_matrix_gmem_m = blockIdx.y * BM + warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            const int store_matrix_gmem_n = blockIdx.x * BN + warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::store_matrix_sync(C+store_matrix_gmem_m * N + store_matrix_gmem_n, C_frag[i][j], N, wmma::mem_row_major);
        }
    }
    return;
}
void hgemm_wmma_m16n16k16_v3(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;

    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_TILE_M * WMMA_M * WARP_TILE_M));
    hgemm_wmma_m16n16k16_kernel_v3<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
    return;
}

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 3;
    if(opt == 1) {
        tester.evaluate(hgemm_wmma_m16n16k16_v1, "hgemm_wmma_m16n16k16_kernel_v1");
    }else if(opt == 2) {
        tester.evaluate(hgemm_wmma_m16n16k16_v2, "hgemm_wmma_m16n16k16_kernel_v2");
    }else if(opt == 3) {
        tester.evaluate(hgemm_wmma_m16n16k16_v3, "hgemm_wmma_m16n16k16_kernel_v3");
    }
    return 0;
}