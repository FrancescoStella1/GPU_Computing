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
#include "./utils/gamma.c"
#include "./utils/CUDA/gamma_gpu.cu"
#include "./utils/gradient.c"
#include "./utils/CUDA/gradient_gpu.cu"
#include "./utils/hog_utils.c"
#include "./utils/CUDA/hog_utils_gpu.cu"


int CUDA_CHECK = 0;     // Temporary
int WRITE = 0;          // Temporary
int CPU = 1;            // Temporary


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

    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n\n", width, height, channels);   

    // Host memory allocation and copy of the loaded image
    unsigned char *h_img = (unsigned char *)malloc(size*channels);     // 3 channels
    unsigned char *h_img_gray = (unsigned char *)malloc(size);
    memcpy(h_img, img, size*3);

    // Grayscale conversion on CPU/GPU
    CPU ? convert(h_img, h_img_gray, size) : cuda_convert(h_img, h_img_gray, width, height);

    if(WRITE)
        stbi_write_jpg("images/results/testGrayScale.jpg", width, height, 1, h_img_gray, 100);

    struct Histogram *hist = createHistogram();

    // Gamma correction on CPU/GPU
    CPU ? gamma_correction(hist, h_img_gray, size) : cuda_gamma_correction(h_img_gray, size);

    if(WRITE)
        stbi_write_jpg("images/results/testGammaCorrection.jpg", width, height, 1, h_img_gray, 100);

    unsigned char* gradientX = (unsigned char*) malloc (size);
    unsigned char* gradientY = (unsigned char*) malloc (size);

    if(CPU) {
        convolutionHorizontal(h_img_gray, gradientX, height, width);
        convolutionVertical(h_img_gray, gradientY, height, width);
    }
    else {
        cuda_compute_gradients(h_img_gray, gradientX, gradientY, width, height);
    }

    if(WRITE) {
        stbi_write_jpg("images/results/gradientX.jpg", width, height, 1, gradientX, 100);
        stbi_write_jpg("images/results/gradientY.jpg", width, height, 1, gradientY, 100);
    }

    unsigned char *magnitude = (unsigned char *)malloc(size);
    unsigned char *direction = (unsigned char *)malloc(size);

    if(CPU) {
        compute_magnitude(gradientX, gradientY, magnitude, width*height);
        compute_direction(gradientX, gradientY, direction, width*height);
    }
    else {
        // Insert cuda functions
    }

    if(WRITE) {
        stbi_write_jpg("images/results/magnitude.jpg", width, height, 1, magnitude, 100);
        stbi_write_jpg("images/results/direction.jpg", width, height, 1, direction, 100);
    }


    if(CPU) {
        compute_hog(magnitude, direction, width, height);
    }

    printf("\n\n [DONE] \n\n");
}
