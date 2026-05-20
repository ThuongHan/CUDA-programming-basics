#include "gpu_utils.h"
#include <chrono>
using namespace std::chrono;

const int BASE = 10;
const int N = 1000000;
const int nr_digits = 3;

#define imin(a,b) (a<b ? a:b)

const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

vector<int> get_random_vector_int(int n) {
    vector<int> v(n);
    for (int i = 0; i < n; i++)
        v[i] = static_cast<int>(rand() % 1000);
    return v;
}

ostream& operator <<(ostream& outStream, const vector<int> v)
{
    for (size_t i = 0; i < v.size(); i++)
        outStream << v[i] << " ";
    return outStream;
}

// GPU kernel that builds a histogram for one radix digit
__global__ void hist(int *data, int *counts, int N, int exp)
{
    __shared__ int cache[BASE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < BASE)
        cache[threadIdx.x] = 0;
    __syncthreads();

    while (tid < N)
    {
        int bucket = (data[tid] / exp) % BASE;
        atomicAdd(&cache[bucket], 1);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BASE; i += blockDim.x)
        atomicAdd(&counts[i], cache[i]);
}
// Convert bucket counts into starting positions for each
void scan(int *counts)
{
    int total = 0;
    for (int i = 0; i < BASE; i++)
    {
        int current = counts[i];
        counts[i] = total;  
        total += current;
    }
}

// Redistributes elements into the correct positions
void distribute_cpu(vector<int>& data, vector<int>& output, int* pos, int N, int exp)
{
    for (int i = 0; i < N; i++)
    {
        int bucket = (data[i] / exp) % BASE;
        output[pos[bucket]++] = data[i];
    }
}
void swap(int*& a, int*& b)
{
    int* temp = a;
    a = b;
    b = temp;
}

// For each digit:
//   1. Compute histogram on GPU
//   2. Scan histogram on CPU
//   3. Redistribute elements
vector<int> radix_sort(vector<int>& h_data, int nr_digits)
{
    int *d_data, *d_counts;

    HANDLE_ERROR( cudaMalloc((void**)&d_data,   N    * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&d_counts, BASE * sizeof(int)) );

    HANDLE_ERROR( cudaMemcpy(d_data, h_data.data(), N * sizeof(int),
                             cudaMemcpyHostToDevice) );

    vector<int> h_input(h_data);
    vector<int> h_output(N);

    for (int d = 0; d < nr_digits; d++)
    {
        int exp = static_cast<int>(pow(BASE, d));

        HANDLE_ERROR( cudaMemset(d_counts, 0, BASE * sizeof(int)) );

        hist<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_counts, N, exp);
        HANDLE_ERROR( cudaGetLastError() );
        HANDLE_ERROR( cudaDeviceSynchronize() );

        int h_counts[BASE];
        HANDLE_ERROR( cudaMemcpy(h_counts, d_counts, BASE * sizeof(int),
                                 cudaMemcpyDeviceToHost) );
        scan(h_counts);

        distribute_cpu(h_input, h_output, h_counts, N, exp);

        swap(h_input, h_output);
        HANDLE_ERROR( cudaMemcpy(d_data, h_input.data(), N * sizeof(int),
                                 cudaMemcpyHostToDevice) );
    }

    HANDLE_ERROR( cudaFree(d_data) );
    HANDLE_ERROR( cudaFree(d_counts) );

    return h_input;
}
int main()
{
    vector<int> h_v = get_random_vector_int(N);
    cout << "Unsorted: " << h_v << endl;

    auto start = high_resolution_clock::now();
    vector<int> h_v_new = radix_sort(h_v, nr_digits);
    auto stop  = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Sorted: " << h_v_new << endl;
    cout << "Execution time " << duration.count() << " milliseconds" << endl;
    return 0;
}

// (GPU) Execution time 390 milliseconds

