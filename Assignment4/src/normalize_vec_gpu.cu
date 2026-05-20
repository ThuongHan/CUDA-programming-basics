#include "gpu_utils.h"
#include <chrono>
using namespace std::chrono;

#define imin(a, b) (a<b ? a:b)

const int N = 1000000;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock);


__global__ void square_length_reduction(const float *d_v, float *d_partial_sums, int N)
{
    extern __shared__ float cache[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) 
    { 
        temp += d_v[tid] * d_v[tid]; 
        tid += blockDim.x * gridDim.x;
    }

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
    
    if (cacheIndex == 0) { d_partial_sums[blockIdx.x] = cache[0]; }
}

__global__ void normalize(float *d_v, int length, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N)
    {
        d_v[tid] = d_v[tid] / length;
        tid += blockDim.x * gridDim.x;
    }
}


int main()
{
    auto start = high_resolution_clock::now();  

    vector<float> h_v = get_random_vector(N);
    cout << "v: " << h_v << endl;

    vector<float> h_partial_sums(blocksPerGrid);

    float *d_v, *d_partial_sums;
    
    HANDLE_ERROR( cudaMalloc( (void**)&d_v, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_partial_sums, blocksPerGrid * sizeof(float) ) );

    HANDLE_ERROR( cudaMemcpy( d_v, h_v.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    
    // Vector length
    square_length_reduction<<< blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >>>( d_v, d_partial_sums, N );
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() ); 

    HANDLE_ERROR( cudaMemcpy( h_partial_sums.data(), 
                              d_partial_sums, 
                              blocksPerGrid * sizeof(float), 
                              cudaMemcpyDeviceToHost) );
    
    float total_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        total_sum += h_partial_sums[i];
    }
    float length = sqrt(total_sum);
    cout << "length before normalization: " << length << endl;

    // Normalize
    normalize<<< blocksPerGrid, threadsPerBlock >>>( d_v, length, N );
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Legnth of a normalized vector
    square_length_reduction<<< blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >>>( d_v, d_partial_sums, N );
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( h_partial_sums.data(), 
                              d_partial_sums, 
                              blocksPerGrid * sizeof(float), 
                              cudaMemcpyDeviceToHost) );

    total_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        total_sum += h_partial_sums[i];
    }
    length = sqrt(total_sum);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "length after normalization: " << length << endl;

    cout << "Execution time " << duration.count() << " milliseconds" << endl;

    return 0;
}

// (CPU) Execution time: 5796 milliseconds 
// (GPU) Execution time 804 milliseconds






