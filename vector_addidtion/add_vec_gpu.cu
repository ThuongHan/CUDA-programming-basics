#include "gpu_utils.h"
#include <cuda_runtime.h>

__global__ void add_vec_gpu(const float* d_v1, const float* d_v2, float* d_add, int n)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
        { d_add[i] = d_v1[i] + d_v2[i]; }
}

int main( void )
{
    const int n = (33 * 1024);

    vector<float> h_v1 = get_random_vector(n);
    vector<float> h_v2 = get_random_vector(n);
    vector<float> h_add(n);

    float *d_v1, *d_v2, *d_add;

    // Allocate device memory
    HANDLE_ERROR( cudaMalloc( (void**) &d_v1, sizeof(float) * n) );
    HANDLE_ERROR( cudaMalloc( (void**) &d_v2, sizeof(float) * n) );
    HANDLE_ERROR( cudaMalloc( (void**) &d_add, sizeof(float) * n) );

    // Transfer data from host to device memory
    HANDLE_ERROR( cudaMemcpy(d_v1, h_v1.data(), sizeof(float) * n, 
                             cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_v2, h_v2.data(), sizeof(float) * n, 
                             cudaMemcpyHostToDevice) );

    const int threads_per_block = 4;
    const int nr_blocks = (n + threads_per_block - 1) / threads_per_block;
    add_vec_gpu<<<nr_blocks, threads_per_block>>>(d_v1, d_v2, d_add, n);    // 4 threads
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Transfer data back to host memory
    HANDLE_ERROR( cudaMemcpy(h_add.data(), d_add, sizeof(float) * n,
                             cudaMemcpyDeviceToHost) );
    
    // Deallocate device memory
    HANDLE_ERROR( cudaFree(d_v1) );
    HANDLE_ERROR( cudaFree(d_v2) );
    HANDLE_ERROR( cudaFree(d_add) );

    cout << "v1: " << h_v1 << endl;
    cout << "v2: " << h_v2 << endl;
    cout << "addition: " << h_add << endl;

    return 0;
}

