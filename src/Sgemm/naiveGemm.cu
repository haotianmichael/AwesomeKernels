#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

/*
A. 线程角度：
    1. Global Mem加载阶段
        thread(5, 7)分别加载相应的A和B矩阵中一个元素——>共2个元素
        tileA[5][7] = A[5行, tile*16+7列]
        tileB[5][7] = B[tile*16+5行, 7列]
    2. Tile阶段
        thread[5][7]需要读取共32个元素:
            tileA[5][0], tileA[5][1]...tileA[5][15]
            tileB[0][7], tileB[1][7]...tileB[15][7]
    3. 写入
        thread[5][7]最终写入C[5][7]

B. 数据角度:
    1. tileA[5][7]这个数据
        * 由线程(threadIdx.y = 5, threadIdx.x=7)加载到缓存
        * 被thread[5][0], thread[5][1]...thread[5][15]这16个线程读取16次
    2. tileB[5][7]这个数据
        * 由线程(threadIdx.y = 5, threadIdx.x=7)加载到缓存
        * 被thread[0][7],thread[1][7]...thread[15][7]这16个线程读取16次
    
C. 数据重用率
    从全局内存加载: 256个线程 * 2个矩阵 = 512次
    从缓存加载: 256个线程 * 16次循环 * 2个矩阵 = 8192次
    数据重用比: 8192 / 512 = 16
    
D. 计算访存比(总FLOP / 从全局内存读写的总字节数)
        总计算量: 256 ^3 x 2 = 33554432 FLOP  (每个元素需要256次乘加)
    1. Naive版本
        全局内存访问:
            每个线程读A的一行: 256个float
            每个线程读B的一行: 256个float
            总共256x256个线程
            总访问: 256 * 256 * (256+256) * 4Bytes       
        计算访存比: 33554432 / 256 * 256 * (512) * 4 = 0.25 FLOP/Bytes

    2. Tiled版本
            A矩阵读一次: 256 * 256 * 4 Bytes
            B矩阵读一次: 256 * 256 * 4 Bytes
        计算访存比: 33554432 / 256 * 256 * 2 * 4 = 64 FLOP/Bytes
         
    比如GPU参数: 内存带宽:1000GB/s  计算能力: 10TFLOPS
        Naive版本 ： 1000GB/s * 0.25 = 250GFLOPS (只用了2.5%算力)
        Tiled版本: 1000GB/s * 64 = 64TFLOS > 10TFLOS(用满算力/受限于带宽)
*/


const int MATRIX_SIZE = 256;
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int N) {

    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;

    for(int tile = 0; tile < (N+15)/16; tile ++) {
        if(row < N && tile * 16 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * 16 + threadIdx.x];
        }else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col < N && tile * 16 + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        }else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for(int k = 0; k < 16; k ++) {
            result += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < N && col < N) {
        C[row * N + col] = result;
    }
}

void initializeMatrix(float *matrix, int size) {
    for(int i = 0; i < size * size; i ++) {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

void printMatrix(const float *matrix, int size) {
    for(int i = 0; i < size; i ++) {
        for(int j = 0; j < size; j ++) {
            std::cout << matrix[i * size + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void printMatrixTopLeft(const float *matrix, int size, int maxRows = 8, int maxCols = 8) {
    std::cout << "\nTop-left " << std::min(maxRows, size) << "x" 
              << std::min(maxCols, size) << " of matrix:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    for (int i = 0; i < std::min(maxRows, size); i++) {
        for (int j = 0; j < std::min(maxCols, size); j++) {
            std::cout << std::setw(8) << matrix[i * size + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {

    int N = MATRIX_SIZE;
    float *hostA = new float[N*N];
    float *hostB = new float[N*N];
    float *hostC = new float[N*N];
    initializeMatrix(hostA, N);
    initializeMatrix(hostB, N);

    float *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, N*N*sizeof(float));
    cudaMalloc(&deviceB, N*N*sizeof(float));
    cudaMalloc(&deviceC, N*N*sizeof(float));

    cudaMemcpy(deviceA, hostA, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x, 
                    (N+blockDim.y-1)/blockDim.y);
    matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hostC, deviceC, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "matrixA" << std::endl;
    printMatrixTopLeft(hostA, N);
    std::cout << "matrixB" << std::endl;
    printMatrixTopLeft(hostB, N);
    std::cout << "matrixC" << std::endl;
    printMatrixTopLeft(hostC, N);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete(hostA);
    delete(hostB);
    delete(hostC);
    return 0;
}