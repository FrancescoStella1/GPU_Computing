#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "./stb_image/stb_image.h"
#include "./stb_image/stb_image_write.h"
#include "./utils/grayscale.c"


int CUDA_CHECK = 0;     // Temporary

int main (int argc, char **argv) {
    if(CUDA_CHECK) {
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
        struct cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA Driver/Runtime version: %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    }

    // Load image and convert to grayscale
    int width, height, channels;
    unsigned char *load;

    if (argc > 1)
        load = stbi_load(argv[1], &width, &height, &channels, 0);
    else
        load = stbi_load("images/calciatore.jpg", &width, &height, &channels, 0);
    if (load == NULL){
        printf("Error loading the image... \n");
        exit(EXIT_FAILURE);
    }
    if(channels<3) {
        printf("Image should have 3 channels.\n");
        exit(EXIT_FAILURE);
    }

    size_t size = width * height;
    unsigned char *gray = calloc(width*height, sizeof(load));
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n\n", width, height, channels);   
    convert(load, gray, size);
    stbi_write_jpg("images/results/testGrayScaleCPU.jpg", width, height, 1, gray, 100);   // 100 
}
