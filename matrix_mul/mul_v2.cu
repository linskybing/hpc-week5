#include <cuda_runtime.h>
#include <iostream>

#define TILE_DIM 32

__global__ void coalescedMultiply(float *a, float* b, float *c,
                                 int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    if(row < N && col < N) {
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
        }
        c[row*N+col] = sum;
    }
}

int main()
{
    int N = 10240;
    size_t sizeA = N * TILE_DIM * sizeof(float);
    size_t sizeB = TILE_DIM * N * sizeof(float);
    size_t sizeC = N * N * sizeof(float);

    float *h_a = new float[N*TILE_DIM];
    float *h_b = new float[TILE_DIM*N];
    float *h_c = new float[N*N];

    for(int i=0; i<N*TILE_DIM; i++) h_a[i] = 1.0f;
    for(int i=0; i<TILE_DIM*N; i++) h_b[i] = 1.0f;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_c, sizeC);

    cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);

    coalescedMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "c[0] = " << h_c[0] << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
