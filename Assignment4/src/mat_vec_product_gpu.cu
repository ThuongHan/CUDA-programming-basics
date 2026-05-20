#include "gpu_utils.h"
#include <chrono>
using namespace std::chrono;

const int rows = 10000;
const int columns = 1000;

#define imin(a,b) (a<b ? a:b)
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (rows*columns+threadsPerBlock-1) / threadsPerBlock);

__global__ void mat_vec_product(const float *d_mat, 
                                const float *d_vec, 
                                float* d_result,
                                int rows, int columns)
{   
    int r = threadIdx.x + blockIdx.x * blockDim.x;

    while (r < rows)
    {
        d_result[r] = 0;
        for (int c = 0; c < columns; c++)
        {
            d_result[r] += d_mat[c + r * columns] * d_vec[c];
        }
        
        r += blockDim.x * gridDim.x;
    }
}

int main() 
{
    vector<float> h_mat = get_random_vector(rows*columns);
    vector<float> h_vec = get_random_vector(columns);
    vector<float> h_result = get_random_vector(rows);

    float *d_mat, *d_vec, *d_result;
    HANDLE_ERROR( cudaMalloc( (void**)&d_mat, rows*columns * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_vec, columns * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&d_result, rows * sizeof(float) ) );

    HANDLE_ERROR( cudaMemcpy( d_mat, h_mat.data(), rows*columns * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_vec, h_vec.data(), columns * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( d_result, h_result.data(), rows * sizeof(float),
                              cudaMemcpyHostToDevice ) );


    auto start = high_resolution_clock::now();
    mat_vec_product<<< blocksPerGrid, threadsPerBlock >>>( d_mat, d_vec, d_result, rows, columns );
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    HANDLE_ERROR( cudaMemcpy( h_result.data(), d_result, rows * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
    
    HANDLE_ERROR( cudaFree( d_mat ) );
    HANDLE_ERROR( cudaFree( d_vec ) );
    HANDLE_ERROR( cudaFree( d_result ) );

    // cout << "Matrix: " << endl;
    // print_mat(cout, h_mat, rows, columns);
    // cout << "Vector: " << endl;
    // print_vec(cout, h_vec);
    // cout << "Result: " << endl;
    // print_vec(cout, h_result);

    cout << "Execution time " << duration.count() << " milliseconds" << endl;

    return 0;
}

// (CPU)  Execution time 82 milliseconds
// (GPU) Execution time 3 milliseconds
