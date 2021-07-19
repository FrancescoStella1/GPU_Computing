#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>


int main(int argc, char **argv) {
    // Get device count
    int deviceCount = 0;

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(error_id!=cudaSuccess) {
        printf("[FAIL]\n\ncudaGetDeviceCount returned %d\n -> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }
    if(deviceCount==0) {
        printf("No CUDA devices found!\n");
    }
    else if(deviceCount==1){
        printf("Detected %d CUDA device\n", deviceCount);
    }
    else {
        printf("Detected %d CUDA devices\n", deviceCount);
    }

    // Get device properties
    int dev=0, driverVersion=0, runtimeVersion=0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver/Runtime version: %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
}