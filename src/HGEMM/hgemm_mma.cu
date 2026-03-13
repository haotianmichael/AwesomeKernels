#include <cstdint>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/tester.h"
#include "common/util.h"
#include "ptx.cuh"

using namespace nvcuda;
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
} 
__device__ __forceinline__ void ld_st_32bit(void *dst, void *src) {
    *reinterpret_cast<half2 *>(dst) = *reinterpret_cast<half2 *>(src);
}

/*
    @ldmatrix首先搬运sharedmem地址, 然后搬运数据
    @自定义uint32_t寄存器
*/
template<const int MMA_M = 16,
         const int MMA_N = 8,
         const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_kernel_v1(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned int K) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M;
    constexpr int BN = MMA_N;
    constexpr int BK = MMA_K;

    __shared__ half tileA[MMA_M][MMA_K];
    __shared__ half tileB[MMA_K][MMA_N];
    __shared__ half tileC[MMA_M][MMA_N];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id = tid % 32;

    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid % 2) * 8;
    const int load_smem_b_k = tid;
    const int load_smem_b_n = 0;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;
    if(load_gmem_a_m >= M && load_gmem_b_n >= N) return;
   
    uint32_t RC[4] = {0, 0, 0, 0};
    for(int k = 0; k < NUM_K_TILES; k ++) {
        int load_gmem_a_k = k * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        ld_st_128bit(&tileA[load_smem_a_m][load_smem_a_k], &A[load_gmem_a_addr]);
        
        if(lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            ld_st_128bit(&tileB[load_smem_b_k][load_smem_b_n], &B[load_gmem_b_addr]);
        }
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[4];

        ptx::ldmatrix_sync(RA, &tileA[lane_id % 16][(lane_id/16)*8]);
        ptx::ldmatrix_trans_sync(RB, &tileB[lane_id % 16][0]);
        ptx::mma_sync_m16n8k16(RC, RA, RB);
        __syncthreads();
    }

    ld_st_32bit(&tileC[lane_id / 4][(lane_id % 4) * 2], &RC[0]);
    ld_st_32bit(&tileC[lane_id / 4 + 8][((lane_id % 4) * 2)], &RC[1]);

    __syncthreads();
    if(lane_id < MMA_M) {
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        ld_st_128bit(&C[store_gmem_c_addr], &tileC[lane_id][0]);
    }
    return;
}
void hgemm_mma_m16n8k16_v1(half *A, half *B, half *C, unsigned int M, unsigned int N, unsigned K) {

    constexpr int MMA_M = 16; 
    constexpr int MMA_K = 16; 
    constexpr int MMA_N = 8; 

    dim3 block(32);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    hgemm_mma_m16n8k16_kernel_v1<MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, C, M, N, K);
    return;
}

/*
    @bank conflict solved by padding
    @wmma寄存器
*/
__global__ void bank_conflict_solver_kernel_padding(half *A, half *B, half *C) {

    __shared__ half tileA[16][16];
    __shared__ half tileB[16][16];

    /* @padding solver
    __shared__ half tileA[16][16 + 8];
    __shared__ half tileB[16][16 + 8];*/
    __shared__ half tileC[16*16];

    int tx = threadIdx.x;
    ld_st_128bit(&(tileA[tx/2][(tx%2)*8]), A+8*tx);
    ld_st_128bit(&(tileB[tx/2][(tx%2)*8]), B+8*tx);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::load_matrix_sync(a_frag, tileA[0], 16);
    wmma::load_matrix_sync(b_frag, tileB[0], 16);

    /* @padding solver
    wmma::load_matrix_sync(a_frag, tileA[0], 16 + 8);
    wmma::load_matrix_sync(b_frag, tileB[0], 16 + 8);*/

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(tileC, c_frag, 16, wmma::mem_row_major);

    __syncthreads();
    ld_st_128bit(C+8*tx, tileC + 8 *tx);
}
void bank_conflict_solver_padding(half *A, half *B, half*C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;
    dim3 block(32);
    dim3 grid(1);
    bank_conflict_solver_kernel_padding<<<grid, block>>>(A, B, C);
    return;
}

/*
    @bank conflict solved by swizzle
    @wmma寄存器
    S: SShift, right shift the addr for swizzleing
    B: BShift, bits to be swizzled
    M: MBase, bits keep the same

template<uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}
__global__ void bank_conflict_solver_kernel_swizzle(half *A, half *B, half *C) {

    __shared__ half tileA[16*16];
    __shared__ half tileB[16*16];
    __shared__ half tileC[16*16];

    int tx = threadIdx.x;
    uint32_t gAddr = tx * 8;
    auto g2sAddr = swizzle<3, 1, 3>(gAddr);
    ld_st_128bit(tileA + g2sAddr, A+gAddr);
    ld_st_128bit(tileB + g2sAddr, B+gAddr);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    uint32_t rAddr = (tx % 16) * 16 + (tx / 16) * 8;
    auto r2sAddr = swizzle<3, 1, 3>(rAddr);

    ptx::ldmatrix_sync(a_frag.x, tileA + r2sAddr);
    ptx::ldmatrix_trans_sync(b_frag.x, tileB +r2sAddr);

    ptx::mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
    ptx::mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
    ptx::stmatrix_sync(tileC + r2sAddr, c_frag.x);

    ld_st_128bit(C+8*tx, tileC + 8 *tx);
}
void bank_conflict_solver_swizzle(half *A, half *B, half*C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;
    dim3 block(32);
    dim3 grid(1);
    bank_conflict_solver_kernel_padding<<<grid, block>>>(A, B, C);
    return;
}
*/

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 1) {
        tester.evaluate(hgemm_mma_m16n8k16_v1, "hgemm_mma_m16n16k16_kernel_v1");
    }
    return 0;
}