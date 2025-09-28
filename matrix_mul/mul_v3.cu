// mul_check.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#define TILE_DIM 32
#define EPSILON 1e-3f

inline void checkCuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

__global__ void transpose_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int tx = blockIdx.y * TILE_DIM + threadIdx.x;
    int ty = blockIdx.x * TILE_DIM + threadIdx.y;

    if (tx < rows && ty < cols) {
        out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matMul_shared_Bt(const float* __restrict__ A,
                                 const float* __restrict__ B_T,
                                 float*       __restrict__ C,
                                 int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // 0..N-1
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // 0..N-1

    float sum = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_row < K) ? B_T[col * K + b_row] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matMulCPU(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>&       C,
               int N, int K) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

int main() {
    const int N = 1024;
    const int K = TILE_DIM;

    std::cout << "N=" << N << " K=" << K << " TILE=" << TILE_DIM << "\n";

    std::vector<float> h_A(N * K), h_B(K * N), h_C_cpu(N * N), h_C_gpu(N * N);

    srand(12345);
    for (int i = 0; i < N * K; ++i) h_A[i] = float(rand() % 10);
    for (int i = 0; i < K * N; ++i) h_B[i] = float(rand() % 10);

    matMulCPU(h_A, h_B, h_C_cpu, N, K);

    float *d_A = nullptr, *d_B = nullptr, *d_Bt = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, (size_t)N * K * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_Bt, (size_t)N * K * sizeof(float)), "cudaMalloc d_Bt");
    checkCuda(cudaMalloc(&d_C, (size_t)N * N * sizeof(float)), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A.data(), (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(d_B, h_B.data(), (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");

    dim3 tBlock(TILE_DIM, TILE_DIM);
    dim3 tGrid( (N + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM );
    transpose_kernel<<<tGrid, tBlock>>>(d_B, d_Bt, K, N);
    checkCuda(cudaGetLastError(), "transpose launch");
    checkCuda(cudaDeviceSynchronize(), "transpose sync");

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid( (N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM );
    matMul_shared_Bt<<<grid, block>>>(d_A, d_Bt, d_C, N, K);
    checkCuda(cudaGetLastError(), "matmul launch");
    checkCuda(cudaDeviceSynchronize(), "matmul sync");

    checkCuda(cudaMemcpy(h_C_gpu.data(), d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy C");

    std::cout << "First 10 C_cpu: ";
    for (int i = 0; i < 10 && i < N*N; ++i) std::cout << h_C_cpu[i] << " ";
    std::cout << "\n";

    std::cout << "First 10 C_gpu: ";
    for (int i = 0; i < 10 && i < N*N; ++i) std::cout << h_C_gpu[i] << " ";
    std::cout << "\n";

    int errors = 0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        float a = h_C_cpu[i];
        float b = h_C_gpu[i];
        float diff = fabs(a - b);
        if (diff > EPSILON) {
            if (errors < 10) {
                std::cout << "Mismatch idx=" << i << " CPU=" << a << " GPU=" << b << " diff=" << diff << "\n";
            }
            ++errors;
        }
    }

    if (errors == 0) std::cout << "OK: GPU matches CPU\n";
    else std::cout << "Total mismatches: " << errors << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Bt);
    cudaFree(d_C);

    return 0;
}
