#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"
#define TILE_SIZE 32

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    // Block computes a TILE_SIZE x TILE_SIZE region of C
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // Thread's position within the block (16x16 layout)
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    
    // This thread computes a 2x2 region starting at:
    int row = block_row * TILE_SIZE + ty * 2;
    int col = block_col * TILE_SIZE + tx * 2;
    
    // Accumulators in registers (2x2 for this thread)
    float C_acc[2][2] = {0};
    
    // Shared memory for tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Loop over K dimension in tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {

        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 2; ++j) {
                int global_row = block_row * TILE_SIZE + ty * 2 + i;
                int global_col = t * TILE_SIZE + tx * 2 + j;
                int local_row = ty * 2 + i;
                int local_col = tx * 2 + j;

                 if (global_row < M && global_col < K) {
                    tile_A[local_row][local_col] = A[global_row * K + global_col];
                 } else {
                     tile_A[local_row][local_col] = 0.0f;
                }
            }
        }
        
        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 2; ++j) {
                int global_row = t * TILE_SIZE + ty * 2 + i;
                int global_col = block_col * TILE_SIZE + tx * 2 + j;
                int local_row = ty * 2 + i;
                int local_col = tx * 2 + j;
        
                if (global_row < K && global_col < N) {
                    tile_B[local_row][local_col] = B[global_row * N + global_col];
                } else {
                    tile_B[local_row][local_col] = 0.0f;
                }
            }       
        }
        
        __syncthreads();
        
        for (int k  = 0; k < TILE_SIZE; ++k){
            C_acc[0][0] += tile_A[ty * 2][k] * tile_B[k][tx*2];
            C_acc[0][1] += tile_A[ty * 2][k] * tile_B[k][tx*2 + 1];
            C_acc[1][1] += tile_A[ty * 2 + 1][k] * tile_B[k][tx*2 + 1];
            C_acc[1][0] += tile_A[ty * 2 + 1][k] * tile_B[k][tx*2];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        if (row < M && col < N) C[(row) * N + (col)] = C_acc[0][0];
        if (row < M && col + 1 < N) C[(row) * N + (col + 1)] = C_acc[0][1];
        if (row + 1 < M && col < N) C[(row + 1) * N + (col)] = C_acc[1][0];
        if (row + 1 < M && col + 1 < N) C[(row + 1) * N + (col + 1)] = C_acc[1][1];
}
}

void matmul_cuda(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(256);
    
    matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_KERNEL_CHECK();
}