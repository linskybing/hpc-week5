#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define TILE_DIM 32
#define EPSILON 1e-3f
#define ceil(a, b) ((a + b - 1) / b)

__global__ void matMulShared(const float* A, const float* B, float* C, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < ceil(K, TILE_DIM); t++) {
        int tiledCol = t * TILE_DIM + threadIdx.x;
        int tiledRow = t * TILE_DIM + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        for(int i=0; i<TILE_DIM; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if(row < N && col < N)
        C[row * N + col] = sum;
}

void matMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N, int K) {
    for(int i = 0; i < N;i++)
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    const int N = 1024;
    const int K = TILE_DIM;

    std::vector<float> h_A(N * K), h_B(K * N), h_C_cpu(N * N), h_C_gpu(N * N);
    for(int i = 0; i< N * K; i++) h_A[i] = static_cast<float>(rand() % 10);
    for(int i=0; i < K * N; i++) h_B[i] = static_cast<float>(rand() % 10);

    matMulCPU(h_A, h_B, h_C_cpu, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * K *sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(ceil(N, TILE_DIM), ceil(N, TILE_DIM));

    matMulShared<<<grid, block>>>(d_A, d_B, d_C, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    int error_count = 0;
    for(int i = 0; i < N * N; i++) {
        float diff = fabs(h_C_gpu[i]-h_C_cpu[i]);
        float denom = fabs(h_C_cpu[i])>1e-6f ? fabs(h_C_cpu[i]) : 1.0f;
        if(diff / denom > EPSILON / 1000) {
            if(error_count<10)
                std::cout << "Mismatch at " << i 
                          << ": GPU=" << h_C_gpu[i] 
                          << ", CPU=" << h_C_cpu[i] << "\n";
            error_count++;
        }
    }

    if(error_count == 0)
        std::cout << "GPU result matches CPU reference.\n";
    else
        std::cout << "Total mismatches: " << error_count << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
