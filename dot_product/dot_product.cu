#include "cuda_runtime.h"
#include "gpu_utils.h"
#include "utils.h"
#include <vector>
using namespace std;

#define imin(a, b) (a<b ? a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot( float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set cache values
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) { c[blockIdx.x] = cache[0]; }
}



int main( void )
{
    // Host vectors (CPU memory)
    vector<float> a = get_random_vector(N), b = get_random_vector(N);
    vector<float> partial_c(blocksPerGrid);

    // Device pointers (GPU memory)
    float *dev_a, *dev_b, *dev_partial_c;

    // Allocate memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c, blocksPerGrid * sizeof(float) ) );

    // Copy data from CPU to GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice ) );   
                              
    dot<<< blocksPerGrid, threadsPerBlock >>>( dev_a, dev_b, dev_partial_c );
    HANDLE_ERROR( cudaGetLastError() ); // Check if kernel launch succeeded
    HANDLE_ERROR( cudaDeviceSynchronize() ); // Wait for GPU to finish execution

    // Copy partial results from GPU to CPU
    HANDLE_ERROR( cudaMemcpy( partial_c.data(), dev_partial_c, 
                              blocksPerGrid * sizeof(float),
                              cudaMemcpyDeviceToHost ) );

    // Final reduction on CPU
    float c = 0;
    for (int i = 0; i < blocksPerGrid; i++) { c += partial_c[i]; }
    
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_partial_c ) );

    cout << "Vector a: " << a << endl;
    cout << "Vector b: " << b << endl;
    cout << "Dot product: " << c << endl;
}



