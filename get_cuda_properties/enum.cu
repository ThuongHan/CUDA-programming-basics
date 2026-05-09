#include <iostream>
#include "gpu_utils.h"

using namespace std;

int main()
{
    int count = get_cuda_device_count();
    cout << "Number of CUDA devices: " << count << endl;

    for (int i = 0; i < count; i++)
    {
        cout << "DEVICE" << i << ":" << endl;
        auto prop = get_cuda_device_properties(i);
        print_cuda_device_properties(prop);
    }
}