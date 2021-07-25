#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "./stb_image/stb_image.h"
#include "./stb_image/stb_image_write.h"
#include "./utils/grayscale.c"
#include "./utils/CUDA/grayscale_gpu.cu"


int CUDA_CHECK = 1;     // Temporary


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
    unsigned char *img;

    if (argc > 1)
        img = stbi_load(argv[1], &width, &height, &channels, 0);
    else
        img = stbi_load("images/calciatore.jpg", &width, &height, &channels, 0);
    if (img == NULL){
        printf("Error loading the image... \n");
        exit(EXIT_FAILURE);
    }
    if(channels<3) {
        printf("Image should have 3 channels.\n");
        exit(EXIT_FAILURE);
    }

    size_t size = width * height * sizeof(unsigned char);
    //printf("Size of the image: %zu\n\n", size);

    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n\n", width, height, channels);   

    // Host memory allocation and copy of the loaded image
    unsigned char *h_img = (unsigned char *)malloc(size*channels);     // 3 channels
    unsigned char *h_img_gray = (unsigned char *)malloc(size);
    memcpy(h_img, img, size*3);

    
    //convert(h_img, h_img_gray, size);
    cuda_convert(h_img, h_img_gray, size);
    stbi_write_jpg("images/results/testGrayScaleCPU.jpg", width, height, 1, h_img_gray, 100);

    // GPU

}