#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gemm_kernel(
    const float* A, 
    const float* B, 
    const float* C, 
    float* D,
    const float alpha, 
    const float beta,
    const int m, 
    const int n, 
    const int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        
        D[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

void init_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 10) / 10.0f;
    }
}

void verify_result(const float* A, const float* B, const float* C, const float* D,
                  float alpha, float beta, int m, int n, int k) {
    float max_error = 0.0f;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float expected = beta * C[i * n + j];
            
            for (int l = 0; l < k; l++) {
                expected += alpha * A[i * k + l] * B[l * n + j];
            }
            
            float error = fabs(D[i * n + j] - expected);
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    
    printf("Maximum error: %e\n", max_error);
    if (max_error < 1e-4) {
        printf("GEMM result is CORRECT\n");
    } else {
        printf("GEMM result is INCORRECT\n");
    }
}

int main() {
    int m = 512;  
    int k = 256;   
    int n = 384;  
    
    float alpha = 2.0f;
    float beta = 3.0f;
    
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    size_t size_D = m * n * sizeof(float);
    
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);
    
    srand(42);
    init_matrix(h_A, m * k);
    init_matrix(h_B, k * n);
    init_matrix(h_C, m * n);
    
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_D, size_D);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("Launching GEMM kernel...\n");
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    
    gemm_kernel<<<numBlocks, threadsPerBlock>>>(
        d_A, d_B, d_C, d_D, alpha, beta, m, n, k
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D, d_D, size_D, cudaMemcpyDeviceToHost);
    
    printf("Verifying result...\n");
    verify_result(h_A, h_B, h_C, h_D, alpha, beta, m, n, k);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    
    printf("GEMM computation completed successfully!\n");
    
    return 0;
}