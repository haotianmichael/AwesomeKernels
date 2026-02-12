#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>


void random_matrix(int m, int n, std::vector<float> &A) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 2.0f);
    for(int i = 0; i < m; i ++)
        for(int j = 0; j < n; j ++) {
            A[i * n + j] = dist(rng);
        }
}
void cpu_sgemm(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C, const int m, const int k, const int n) {

    for(int i = 0; i < m; i ++) {
        for(int j = 0; j < n; j ++) {
            float res = 0.0f;
            for(int _k = 0; _k < k; _k++) {
                res += A[i * k + _k] * B[_k * n + j]; 
            }
            C[i * n + j] = res;
        }
    }
}
float compare_matrices(int m, int n, std::vector<float> &gpu_C, std::vector<float> &cpu_C) {
    float max_diff = 0, diff;
    int printed = 0;

    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j ++) {
            diff = std::abs(gpu_C[i * n + j] - cpu_C[i * n + j]);
            max_diff = (diff > max_diff) ? diff : max_diff;
            if(printed == 0) {
                if(max_diff > 0.5f || max_diff < -0.5f) {
                    printf("\n error: i %d j %d\n", i, j);
                    return max_diff;
                }
            }
        }
    std::cout << "right" << std::endl;
    return 0;
}
/*
global_mem: 
    read: 每个线程:2K次，共M*N个线程——2MNK次
    write: MN次
*/
__global__ void cuda_sgemm_v0(float *A, float *B, float *C, const int M, const int N, const int K) {

    float *A_bck = A + blockIdx.x * K * blockDim.x;
    float *B_bck = B + blockIdx.y * blockDim.y;

    float res = 0.0f;
    for(int k = 0; k < K; k++) {
        res += A_bck[threadIdx.x * K + k] * B_bck[threadIdx.y + k * M];
    }
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    C[x * N + y] = res;
    return;
}

int main() {
   
    int m = 512;
    int n = 512;
    int k = 512;
    
    std::vector<float> h_A(m * k), h_B(k * n), h_C(m * n), gpu_C(m * n);

    random_matrix(m, k, h_A);
    random_matrix(k, n, h_B);
    std::fill(h_C.begin(), h_C.end(), 0.0f);
    std::fill(gpu_C.begin(), gpu_C.end(), 0.0f);

    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, m * k * sizeof(float));
    cudaMalloc(&device_B, k * n * sizeof(float));
    cudaMalloc(&device_C, m * n * sizeof(float));

    cudaMemcpy(device_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    

    cpu_sgemm(h_A, h_B, h_C, m, k, n);

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    cuda_sgemm_v0<<<grid,block>>>(device_A, device_B, device_C, m, n, k);
    cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float res = compare_matrices(m, n, gpu_C, h_C); 
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    return 0;
}