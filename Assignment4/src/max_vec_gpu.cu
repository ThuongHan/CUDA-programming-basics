#include "gpu_utils.h"
#include <tuple>
#include <chrono>
using namespace std::chrono;

#define imin(a, b) (a<b ? a:b)

const int N = 1000000;   
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32 , (N+threadsPerBlock-1) / threadsPerBlock );


__global__ void max_reduction(const float *d_v, float *blockMaxVal, int *blockMaxIdx, int N)
{
    // cache: [ float[] | int[] ]
    extern __shared__ char cache[];
    float* cache_floats = reinterpret_cast<float*>( cache );
    int* cache_ints = reinterpret_cast<int*>( cache + blockDim.x * sizeof(float) );

    // Local max for each thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float max_el = -FLT_MAX;
    int max_idx = -1;

    while (tid < N)
    {
        if (d_v[tid] > max_el)
        {
            max_el = d_v[tid];
            max_idx = tid;
        }
        tid += blockDim.x * gridDim.x;
    }

    cache_floats[threadIdx.x] = max_el;
    cache_ints[threadIdx.x] = max_idx;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            if (cache_floats[threadIdx.x + i] > cache_floats[threadIdx.x])
            {
                cache_floats[threadIdx.x] = cache_floats[threadIdx.x + i];
                cache_ints[threadIdx.x] = cache_ints[threadIdx.x + i];
            }
        }
        __syncthreads();

        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        blockMaxVal[blockIdx.x] = cache_floats[0];
        blockMaxIdx[blockIdx.x] = cache_ints[0];
    }
}

int main() 
{
    auto start = high_resolution_clock::now();  

    vector<float> h_v = get_random_vector(N);
    vector<float> h_blockMaxVal(blocksPerGrid);
    vector<int> h_blockMaxIdx(blocksPerGrid);

    float *d_v, *d_blockMaxVal;
    int *d_blockMaxIdx;

    // Allocate memory on GPU
    HANDLE_ERROR( cudaMalloc( (void**)&d_v, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_blockMaxVal, blocksPerGrid * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_blockMaxIdx, blocksPerGrid * sizeof(int) ) );

    // Copy data from CPU to GPU
    HANDLE_ERROR( cudaMemcpy( d_v, h_v.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice ) );

    max_reduction<<< blocksPerGrid, threadsPerBlock, 
                     threadsPerBlock * sizeof(float) +
                     threadsPerBlock * sizeof(int) >>>( d_v, d_blockMaxVal, d_blockMaxIdx, N);
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Copy data from GPU to CPU
    HANDLE_ERROR( cudaMemcpy( h_blockMaxVal.data(), d_blockMaxVal, blocksPerGrid * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( h_blockMaxIdx.data(), d_blockMaxIdx, blocksPerGrid * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // Retrieve max value and index from the reduced array

    float max_el = -FLT_MAX;
    int max_idx = -1;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        if (h_blockMaxVal[i] > max_el)
        {
            max_el = h_blockMaxVal[i];
            max_idx = h_blockMaxIdx[i];
        }
    }

    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<milliseconds>(stop - start);

    HANDLE_ERROR( cudaFree( d_v ) );
    HANDLE_ERROR( cudaFree( d_blockMaxVal) );
    HANDLE_ERROR( cudaFree( d_blockMaxIdx ) );

    cout << "v: " << h_v << endl;
    cout << "Max element: " << max_el << "at index: " << max_idx << endl;
    cout << "Execution time " << duration.count() << " milliseconds" << endl;

    return 0;
}

// Execution time 409 milliseconds