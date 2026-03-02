#include <cuda_runtime.h>
#include "common/tester.h"

using namespace nvcuda;
/*
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void store_matrix_sync(fragment<...> &a, const T *mprt, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &a, fragment<...> &b, const fragment<...> &c, bool staf=false);
*/

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

    for(int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(A_frag, A_bck + k, K);
        wmma::load_matrix_sync(B_frag, B_bck + k * N, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
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

int main() {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    const int opt = 1;
    if(opt == 1) {
        tester.evaluate(hgemm_wmma_m16n16k16_v1, "hgemm_wmma_m16n16k16_kernel");
    }else if(opt == 2) {

    }
    return 0;
}