#ifndef __GPU_UTILS_H__
#define __GPU_UTILS_H__

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "utils.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int get_cuda_device_count() {
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    return count;
}

cudaDeviceProp get_cuda_device_properties(int device_id) {
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

void print_cuda_device_properties(const cudaDeviceProp& prop) {
    std::cout<< "Device  prop.name: " << prop.name << std::endl;
    std::cout<< "  Compute capability  prop.major: " << prop.major << ", prop.minor: " << prop.minor << std::endl;
    std::cout<< "  Clock rate  prop.clockRate: " << prop.clockRate << std::endl;
    std::cout<< "  Multiprocessor count  prop.multiProcessorCount: " << prop.multiProcessorCount << std::endl;
    std::cout<< "  Total global memory  prop.totalGlobalMem: " << prop.totalGlobalMem << " bytes" << std::endl;
    std::cout<< "  Total constant memory  prop.totalConstMem: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout<< "  Shared memory per block  prop.sharedMemPerBlock: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout<< "  Shared memory per multiprocessor  prop.sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout<< "  Registers per block  prop.regsPerBlock: " << prop.regsPerBlock << std::endl;
    std::cout<< "  Registers per multiprocessor  prop.regsPerMultiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout<< "  Warp size  prop.warpSize: " << prop.warpSize << std::endl;
    std::cout<< "  Max threads per block  prop.maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout<< "  Max threads per multiprocessor  prop.maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout<< "  Max thread dimensions  prop.maxThreadsDim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout<< "  Max grid dimensions  prop.maxGridSize: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout<< "  Concurrent kernels  prop.concurrentKernels: " << prop.concurrentKernels << std::endl;
    std::cout<< "  Async engine count  prop.asyncEngineCount: " << prop.asyncEngineCount << std::endl;
    std::cout<< "  Kernel execution timeout enabled  prop.kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << std::endl;
}


#endif