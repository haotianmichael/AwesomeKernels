#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <pthread.h>
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
                    printf("\n error: i %d j %d\ncpu: %f gpu: %f\n", i, j, cpu_C[i*n+j], gpu_C[i*n+j]);
                    return max_diff;
                }
            }
        }
    std::cout << "right" << std::endl;
    return 0;
}
/*
navie global_mem: 
    read: 每个线程:2K次，共M*N个线程——2MNK次
    write: MN次
*/
__global__ void cuda_sgemm_v0(float *A, float *B, float *C, const int M, const int N, const int K) {

    float *A_bck = A + blockIdx.x * K * blockDim.x;
    float *B_bck = B + blockIdx.y * blockDim.y;

    float res = 0.0f;
    for(int k = 0; k < K; k++) {
        res += A_bck[threadIdx.x * K + k] * B_bck[threadIdx.y + k * N];
    }
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    C[x * N + y] = res;
    return;
}

/*
shared_mem的过渡版本:
    cost too much shared_mem per block;
*/
template<unsigned int BLOCK_SIZE, unsigned int K_>
__global__ void cuda_sgemm_v1(float *A, float *B, float *C, const int M, const int N, const int K) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    float *A_bck = A + blockDim.x * blockIdx.x * K;
    float *B_bck = B + blockDim.y * blockIdx.y;

    __shared__ float a_shared[BLOCK_SIZE][K_];
    __shared__ float b_shared[K_][BLOCK_SIZE];

    for(int s = 0; s < K; s += blockDim.x) {
        a_shared[threadIdx.x][threadIdx.y + s] = A_bck[threadIdx.x * K + threadIdx.y + s];
        b_shared[threadIdx.x + s][threadIdx.y] = B_bck[(threadIdx.x + s)* N + threadIdx.y];        
    }
    __syncthreads();

    float res = 0.0f;
    for(int k = 0; k < K; k ++) {
        //res += A_bck[threadIdx.x * K + k] * B_bck[threadIdx.y + N * k];
        res += a_shared[threadIdx.x][k] * b_shared[k][threadIdx.y];
    }
    C[x * N + y] = res;
}

/*
v2: tiled shared_mem
*/
template<unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm_v2(float *A, float *B, float *C, const int M, const int N, const int K) {

    float *A_bck = A + blockIdx.x * blockDim.x * K;
    float *B_bck = B + blockIdx.y * blockDim.y;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    float res = 0.0f;
    for(int s = 0; s < K; s += BLOCK_SIZE) {
        a_shared[threadIdx.x][threadIdx.y] = A_bck[threadIdx.x * K + threadIdx.y + s];
        b_shared[threadIdx.x][threadIdx.y] = B_bck[(threadIdx.x + s) * N + threadIdx.y];

        __syncthreads();
        for(int k = 0; k < BLOCK_SIZE; k ++) {
            res += a_shared[threadIdx.x][k] * b_shared[k][threadIdx.y];
        }
        __syncthreads();
    }
    C[x * N + y] = res;

    return;
}

/*
向量化加载的过度版本
*/
template<unsigned int BLOCK_SIZE, unsigned int STRIDE> 
__global__ void cuda_sgemm_v3(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int STEP = BLOCK_SIZE * STRIDE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    float *A_bck = A + blockIdx.y * STEP * K;
    float *B_bck = B + blockIdx.x * STEP;

    __shared__ float tileA[STEP][STEP];
    __shared__ float tileB[STEP][STEP];

    float res[STRIDE][STRIDE];
    for(int i = 0; i < STRIDE; i ++)
        for(int j = 0; j < STRIDE; j ++)
            res[i][j] = 0;

    for(int s = 0; s < K; s += STEP) {
        for(int i = 0; i < STRIDE; i ++) {
            for(int j = 0; j < STRIDE; j ++) {
                 tileA[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A_bck[(ty + i * BLOCK_SIZE)*K + tx + j * BLOCK_SIZE + s];
                 tileB[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B_bck[(ty + i * BLOCK_SIZE + s) * N + tx + j * BLOCK_SIZE];
            }
        }
        __syncthreads();

        for(int i = 0; i < STRIDE; i ++) {
            for(int j = 0; j < STRIDE; j ++) {
                for(int k = 0; k < STEP; k ++) {
                    res[i][j] += tileA[ty + i * BLOCK_SIZE][k] * tileB[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }
    float *C_bck = C + blockIdx.y * STEP * N + blockIdx.x * STEP;
    for(int i = 0; i < STRIDE; i ++) {
        for(int j = 0; j < STRIDE; j ++) {
            C_bck[(ty + i * BLOCK_SIZE) * N + tx + j * BLOCK_SIZE] = res[i][j];
        }
    }

}

/*
v4: 向量化加载+float4
*/
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_v4(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    float *A_bck = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float res[NUM_PER_THREAD];
    for(int i = 0; i < NUM_PER_THREAD; i ++) res[i] = 0.0f;

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        FETCH_FLOAT4(tileA[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_bck[ty * K + s + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(tileB[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_bck[(ty + s) * N + tx * NUM_PER_THREAD]);

        __syncthreads();
        for(int i = 0; i < NUM_PER_THREAD; i ++) {
            for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
                res[i] += tileA[ty][k] * tileB[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < NUM_PER_THREAD; i ++) {
        C_bck[ty * N + tx * NUM_PER_THREAD + i] = res[i];
    }
    return;
}

/*
寄存器分块 + 外积 + 索引重排的过渡版本
初始线程布局: 8 x 32
在做计算的时候索引重排
*/
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_v5(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ctx = tid % 16;
    const int cty = tid / 16;
    float *A_bck = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    const int REG_NUM = NUM_PER_THREAD / 2;
    float regA[REG_NUM] = {0.0f};
    float regB[REG_NUM] = {0.0f};
    float res[REG_NUM][REG_NUM] = {0.0f};

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK){
        FETCH_FLOAT4(tileA[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_bck[ty * K + s + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(tileB[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_bck[(ty + s) * N + tx * NUM_PER_THREAD]);
        __syncthreads();
        for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
            regA[0] = tileA[cty * 2][k];
            regA[1] = tileA[cty * 2 + 1][k];
            regB[0] = tileB[k][ctx * 2];
            regB[1] = tileB[k][ctx * 2 + 1];
            for(int i = 0; i < REG_NUM; i ++) {
                for(int j = 0; j < REG_NUM; j ++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }
    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < REG_NUM; i ++) {
        for(int j = 0; j < REG_NUM; j ++) {
            C_bck[(cty * 2 + i) * N + ctx * 2 + j] = res[i][j];
        }
    }
    return;
}

/*
初始线程布局: 16x16
做加载的时候做索引重排 */
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm_v5_(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * blockDim.x + tx;
    const int ctx_load = tid % 8;
    const int cty_load = tid / 8;
    float *A_bck = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;
    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    
    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    const int REG_NUM = NUM_PER_THREAD / 2;
    float res[REG_NUM][REG_NUM] = {0.0f};
    float regA[REG_NUM];
    float regB[REG_NUM];

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {

        FETCH_FLOAT4(tileA[cty_load][ctx_load * NUM_PER_THREAD]) = FETCH_FLOAT4(A_bck[cty_load * K + s + ctx_load * NUM_PER_THREAD]);
        FETCH_FLOAT4(tileB[cty_load][ctx_load * NUM_PER_THREAD]) = FETCH_FLOAT4(B_bck[(cty_load + s) * N + ctx_load * NUM_PER_THREAD]);

        __syncthreads();
        /* 普通内积版本
        for(int i = 0; i < REG_NUM; i ++) {
            for(int j = 0; j < REG_NUM; j ++) {
                for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
                    res[i][j] += tileA[ty * REG_NUM + i][k] * tileB[k][tx * REG_NUM + j];
                }
            }
        }*/
        //寄存器分块+外积版本
        for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
            regA[0] = tileA[ty * REG_NUM][k];
            regA[1] = tileA[ty * REG_NUM + 1][k];
            regB[0] = tileB[k][tx * REG_NUM];
            regB[1] = tileB[k][tx * REG_NUM + 1];
            for(int i = 0; i < REG_NUM; i ++) {
                for(int j = 0; j < REG_NUM; j ++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    for(int i = 0; i < REG_NUM; i ++) {
        for(int j = 0; j < REG_NUM; j ++) {
            C_bck[(ty * REG_NUM +i) * N + tx * REG_NUM + j] = res[i][j];
        }
    }
}

/*
寄存器分块+外积+FLOAT4过渡版本
*/
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int M_NUM_PER_THREAD,
         unsigned int N_NUM_PER_THREAD,
         unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_v6(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    float *A_bck = A +  blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];  // 64
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float regA[M_NUM_PER_THREAD] = {0.0f};
    float regB[N_NUM_PER_THREAD] = {0.0f};

    float res[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.0f};
    
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {

        for(int i = 0; i < M_NUM_PER_THREAD; i ++) { // 4
           FETCH_FLOAT4(tileA[ty * M_NUM_PER_THREAD + i][tx * K_NUM_PER_THREAD])  
                = FETCH_FLOAT4(A_bck[K * (ty * M_NUM_PER_THREAD + i) + s + tx * K_NUM_PER_THREAD]);
        }
        for(int j = 0; j < K_NUM_PER_THREAD; j ++) {
            FETCH_FLOAT4(tileB[ty * K_NUM_PER_THREAD + j][tx * N_NUM_PER_THREAD]) 
                = FETCH_FLOAT4(B_bck[N * (ty * K_NUM_PER_THREAD + j + s) + tx * N_NUM_PER_THREAD]);
        }
        __syncthreads(); 
        for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
            regA[0] = tileA[ty * M_NUM_PER_THREAD][k];
            regA[1] = tileA[ty * M_NUM_PER_THREAD + 1][k];
            regA[2] = tileA[ty * M_NUM_PER_THREAD + 2][k];
            regA[3] = tileA[ty * M_NUM_PER_THREAD + 3][k];
            FETCH_FLOAT4(regB[0]) = FETCH_FLOAT4(tileB[k][tx * N_NUM_PER_THREAD]);
            for(int i = 0; i < M_NUM_PER_THREAD; i++) {
                for(int j = 0; j < N_NUM_PER_THREAD; j ++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < M_NUM_PER_THREAD; i ++) {
        FETCH_FLOAT4(C_bck[N * (ty * M_NUM_PER_THREAD + i) + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(res[i][0]);
    }
    return;
}

/*
v6_: FLOAT4 + 一个线程处理4x4个数据
*/
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int M_NUM_PER_THREAD,
         unsigned int N_NUM_PER_THREAD,
         unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_v6_(float *A, float *B, float *C, const int M, const int N, const int K) {

    float *A_bck = A + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;
    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float res[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.0f};

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        for(int i = 0; i < M_NUM_PER_THREAD; i ++) {
            FETCH_FLOAT4(tileA[ty * M_NUM_PER_THREAD + i][tx * K_NUM_PER_THREAD]) = FETCH_FLOAT4(A_bck[K * (ty * M_NUM_PER_THREAD + i) + s + tx * K_NUM_PER_THREAD]);
            FETCH_FLOAT4(tileB[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(B_bck[N * (ty * K_NUM_PER_THREAD + i + s) + tx * N_NUM_PER_THREAD]);
        } 

        __syncthreads();
        for(int i = 0; i < M_NUM_PER_THREAD; i ++) {
            for(int j = 0; j < N_NUM_PER_THREAD; j ++) {
                for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
                    res[i][j] += tileA[ty * M_NUM_PER_THREAD + i][k] * tileB[k][tx * N_NUM_PER_THREAD + j];
                }
            }
        } 
        __syncthreads();
    }
    for(int i = 0; i < M_NUM_PER_THREAD; i ++) {
        for(int j = 0; j < N_NUM_PER_THREAD; j ++) {
            C_bck[N * (i + ty * M_NUM_PER_THREAD) + tx * N_NUM_PER_THREAD + j] = res[i][j];
        }
    }
    return;
}

/*
v7: 寄存器分块 + 外积 + FLOAT4 + 转置
*/
template<unsigned int M_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int M_NUM_PER_THREAD,
         unsigned int N_NUM_PER_THREAD,
         unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm_v7(float *A, float *B, float *C, const int M, const int N, const int K) {

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    float *A_bck = A +  blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float tileA[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];  // 64
    __shared__ float tileB[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float regA[M_NUM_PER_THREAD] = {0.0f};
    float regB[N_NUM_PER_THREAD] = {0.0f};
    float a_load_reg[K_NUM_PER_THREAD] = {0.0f};
    float res[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.0f};
    
    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {

        for(int i = 0; i < M_NUM_PER_THREAD; i ++) { // 4
           FETCH_FLOAT4(a_load_reg[0])  
                = FETCH_FLOAT4(A_bck[K * (ty * M_NUM_PER_THREAD + i) + s + tx * K_NUM_PER_THREAD]);
           tileA[tx * K_NUM_PER_THREAD][ty * M_NUM_PER_THREAD + i] = a_load_reg[0];
           tileA[tx * K_NUM_PER_THREAD + 1][ty * M_NUM_PER_THREAD + i] = a_load_reg[1];
           tileA[tx * K_NUM_PER_THREAD + 2][ty * M_NUM_PER_THREAD + i] = a_load_reg[2];
           tileA[tx * K_NUM_PER_THREAD + 3][ty * M_NUM_PER_THREAD + i] = a_load_reg[3];
        }
        for(int j = 0; j < K_NUM_PER_THREAD; j ++) {
            FETCH_FLOAT4(tileB[ty * K_NUM_PER_THREAD + j][tx * N_NUM_PER_THREAD]) 
                = FETCH_FLOAT4(B_bck[N * (ty * K_NUM_PER_THREAD + j + s) + tx * N_NUM_PER_THREAD]);
        }
        __syncthreads(); 
        for(int k = 0; k < K_NUM_PER_BLOCK; k ++) {
            FETCH_FLOAT4(regA[0]) = FETCH_FLOAT4(tileA[k][ty * N_NUM_PER_THREAD]);
            FETCH_FLOAT4(regB[0]) = FETCH_FLOAT4(tileB[k][tx * N_NUM_PER_THREAD]);
            for(int i = 0; i < M_NUM_PER_THREAD; i++) {
                for(int j = 0; j < N_NUM_PER_THREAD; j ++) {
                    res[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    float *C_bck = C + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < M_NUM_PER_THREAD; i ++) {
        FETCH_FLOAT4(C_bck[N * (ty * M_NUM_PER_THREAD + i) + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(res[i][0]);
    }
    return;
}

/*
v8: 双缓冲
*/
/*template<unsigned int M_NUM_PER_BLOCK,
         unsigned int K_NUM_PER_BLOCK,
         unsigned int N_NUM_PER_BLOCK,
         unsigned int THREAD_SIZE_X,
         unsigned int THREAD_SIZE_Y>
__global__ void cuda_sgemm_v8(float *A, float *B, float *C, const int M, const int N, const int K) {

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = ty * blockDim.x + tx;
        const int cty = tid % 8;  // 0-7
        const int ctx = tid / 8;

        float *A_bck = A + blockIdx.y * M_NUM_PER_BLOCK * K;
        float *B_bck = B + blockIdx.x * N_NUM_PER_BLOCK;

        __shared__ float tileA[THREAD_SIZE_Y][THREAD_SIZE_X];
        __shared__ float tileB[THREAD_SIZE_Y][THREAD_SIZE_X];
        float regA[THREAD_SIZE_Y];
        float regB[THREAD_SIZE_X];
        float res[THREAD_SIZE_Y][THREAD_SIZE_X];

        for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {

            FETCH_FLOAT4(tileA[][cty])
            FETCH_FLOAT4(tileB[cty][ctx * 4]) = FETCH_FLOAT4(B_bck[(cty+s) * N + ctx * 4]);

            __syncthreads();
            for(int i = 0; i < THREAD_SIZE_X; i ++) {
                for(int j = 0; j < THREAD_SIZE_Y; j ++) {
                    res[i][j] += tileA[][] * tileB[][]; 
                }
            }
            __syncthreads();
        }


        return;
}*/

int main() {
   
    int m = 512;
    int n = 512;
    constexpr int k = 256;
    
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
    const int opt = 5;
    constexpr int BLOCK = 16;
    if(opt == 0) {
        std::cout << "cuda_sgemm_v0" << std::endl;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

        cuda_sgemm_v0<<<grid,block>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 1) {
        std::cout << "cuda_sgemm_v1" << std::endl;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

        //cuda_sgemm_v1<BLOCK, k><<<grid, block>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 2) {
        std::cout << "cuda_sgemm_v2" << std::endl;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

        cuda_sgemm_v2<BLOCK><<<grid, block>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 3) {
        std::cout << "cuda_sgemm_v3" << std::endl;
        constexpr int STRIDE = 2;
        dim3 block(BLOCK, BLOCK);
        dim3 grid_v3((m + BLOCK-1)/BLOCK/STRIDE, (m + BLOCK - 1)/BLOCK/STRIDE);

        cuda_sgemm_v3<BLOCK, STRIDE><<<grid_v3, block>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 4) {
        std::cout << "cuda_sgemm_v4" << std::endl;
        constexpr int M_NUM_PER_BLOCK = 32;    
        constexpr int N_NUM_PER_BLOCK = 32;    
        constexpr int K_NUM_PER_BLOCK = 32;    
        constexpr int NUM_PER_THREAD = 4;    

        dim3 block_v4(8, 32);
        dim3 grid_v4((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
        cuda_sgemm_v4<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_v4, block_v4>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 5) {
        std::cout << "cuda_sgemm_v5" << std::endl;
        constexpr int M_NUM_PER_BLOCK = 32;    
        constexpr int N_NUM_PER_BLOCK = 32;    
        constexpr int K_NUM_PER_BLOCK = 32;    
        constexpr int NUM_PER_THREAD = 4;    

        dim3 block_v4(8, 32);
        dim3 block_v4_new(16, 16);
        dim3 grid_v4((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
        //cuda_sgemm_v5<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_v4, block_v4>>>(device_A, device_B, device_C, m, n, k);
        cuda_sgemm_v5_<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid_v4, block_v4_new>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 

    }else if(opt == 6) {
        std::cout << "cuda_sgemm_v6" << std::endl;
        constexpr int M_NUM_PER_BLOCK = 64;    
        constexpr int N_NUM_PER_BLOCK = 64;    
        constexpr int K_NUM_PER_BLOCK = 64;    
        constexpr int M_NUM_PER_THREAD = 4;    
        constexpr int N_NUM_PER_THREAD = 4;    
        constexpr int K_NUM_PER_THREAD = 4;    

        dim3 block_v6(16, 16);
        dim3 grid_v6((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
        cuda_sgemm_v6_<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, M_NUM_PER_THREAD, N_NUM_PER_THREAD, K_NUM_PER_THREAD><<<grid_v6, block_v6>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 7) {
        std::cout << "cuda_sgemm_v7" << std::endl;
        constexpr int M_NUM_PER_BLOCK = 64;    
        constexpr int N_NUM_PER_BLOCK = 64;    
        constexpr int K_NUM_PER_BLOCK = 64;    
        constexpr int M_NUM_PER_THREAD = 4;    
        constexpr int N_NUM_PER_THREAD = 4;    
        constexpr int K_NUM_PER_THREAD = 4;    

        dim3 block_v6(16, 16);
        dim3 grid_v6((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
        cuda_sgemm_v7<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, M_NUM_PER_THREAD, N_NUM_PER_THREAD, K_NUM_PER_THREAD><<<grid_v6, block_v6>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); 
    }else if(opt == 8) {
        std::cout << "cuda_sgemm_v8" << std::endl;
        constexpr int M_NUM_PER_BLOCK = 128;
        constexpr int N_NUM_PER_BLOCK = 128;
        constexpr int K_NUM_PER_BLOCK = 8;
        constexpr int THREAD_SIZE_X = 8;
        constexpr int THREAD_SIZE_Y = 8;

        /*dim3 block_v8(16, 16);
        dim3 grid_v8((m + M_NUM_PER_BLOCK - 1) / M_NUM_PER_BLOCK, (n + N_NUM_PER_BLOCK - 1) / N_NUM_PER_BLOCK);
        cuda_sgemm_v8<M_NUM_PER_BLOCK, K_NUM_PER_BLOCK, N_NUM_PER_BLOCK, THREAD_SIZE_X, THREAD_SIZE_Y><<<grid_v8, block_v8>>>(device_A, device_B, device_C, m, n, k);
        cudaMemcpy(gpu_C.data(), device_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        compare_matrices(m, n, gpu_C, h_C); */
    }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    return 0;
}