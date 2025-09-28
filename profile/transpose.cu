#include <cuda_runtime.h>
#include <iostream>

#define ROWS 8192
#define COLS 4096

__global__ void transpose(float* out, const float* in, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col]; // transpose
    }
}

int main() {
    size_t size_in = ROWS * COLS;
    size_t bytes_in = size_in * sizeof(float);

    // Host allocation
    float* h_in = new float[size_in];
    float* h_out = new float[size_in];

    // Initialize input
    for (size_t i = 0; i < size_in; i++)
        h_in[i] = float(i);

    // Device allocation
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes_in);
    cudaMalloc(&d_out, bytes_in);

    // Copy to GPU
    cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose<<<numBlocks, threadsPerBlock>>>(d_out, d_in, ROWS, COLS);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_out, d_out, bytes_in, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            float expected = h_in[r * COLS + c];
            float got = h_out[c * ROWS + r];  // transpose
            if (fabs(expected - got) > 1e-5) {
                std::cout << "Mismatch at (" << r << "," << c << "): "
                          << got << " != " << expected << "\n";
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct)
        std::cout << "Transpose verification PASSED!\n";
    else
        std::cout << "Transpose verification FAILED!\n";

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}