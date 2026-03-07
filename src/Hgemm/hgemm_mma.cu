#include <cstdint>
#include <cuda_runtime.h>
#include "common/tester.h"
#include "common/util.h"


#define LDST128BITS(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define LDST32BITS(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_t(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))


/*
    @ldmatrix首先搬运sharedmem地址, 然后搬运数据
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

    uint32_t RC[2] = {0, 0};
    for(int k = 0; k < NUM_K_TILES; k ++) {
        int load_gmem_a_k = k * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        LDST128BITS(tileA[load_smem_a_m][load_smem_a_k]) = 
            LDST128BITS(A[load_gmem_a_addr]);
        
        if(lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k;
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            LDST128BITS(tileB[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
        }
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&tileA[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);
        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&tileB[lane_id % 16][0]);
        LDMATRIX_X2_t(RB[0], RB[1], load_smem_b_ptr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        __syncthreads();
    }

    LDST32BITS(tileC[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(tileC[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

    __syncthreads();
    if(lane_id < MMA_M) {
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr]) = LDST128BITS(tileC[lane_id][0]);
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

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 1) {
        tester.evaluate(hgemm_mma_m16n8k16_v1, "hgemm_mma_m16n16k16_kernel_v1");
    }
    return 0;
}