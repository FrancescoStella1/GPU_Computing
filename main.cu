#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
#include "./utils/video_utils.c"
#include "./utils/timing.c"

#define CUDA_CHECK   0
#define WRITE   0
#define CPU   1
#define N_STREAMS   4
#define CPU_TIMING   "timing_cpu.txt"
#define GPU_TIMING   "timing_gpu.txt"


int main (int argc, char **argv) {
    if(CUDA_CHECK) {
        // Device properties
        struct cudaDeviceProp deviceProp;
        int dev=0, driverVersion=0, runtimeVersion=0, maxMultiprocessors=0;

        // Device count
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if(error_id!=cudaSuccess) {
            fprintf(stderr, "[FAIL]\n\ncudaGetDeviceCount returned %d\n -> %s\n", (int)error_id, cudaGetErrorString(error_id));
            exit(EXIT_FAILURE);
        }
        
        if(deviceCount==0) {
            fprintf(stderr, "No CUDA devices found!\n");
            exit(-1);
        }
        else {
            for(int d=0; d<deviceCount; d++) {
                cudaGetDeviceProperties(&deviceProp, d);
                if(maxMultiprocessors < deviceProp.multiProcessorCount) {
                    maxMultiprocessors = deviceProp.multiProcessorCount;
                    dev = d;
                }
            }
        }

        cudaSetDevice(dev);
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA Driver/Runtime version: %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    }

    // Load image and convert to grayscale
    int width, height, channels;
    unsigned char *img;

    if (argc > 2) {
        if(strcmp(argv[1], "-i") == 0)
            img = stbi_load(argv[2], &width, &height, &channels, 0);
        else if(strcmp(argv[1], "-v") == 0) {
            extract_frames(argv[2]);
            printf("Frames extracted\n");
            process_frames("./images/results/frames", CPU, N_STREAMS, WRITE);
            exit(1);
        }
        else {
            printf("\n\nPlease specify two arguments:\n\n- the type of the input file (-i for image and -v for video)\n- the file path\n");
            exit(-1);
        }
    }
    else {
        printf("\n\nPlease specify two arguments:\n\n- the type of the input file (-i for image and -v for video)\n- the file path\n");
        exit(-1);
    }
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
    if(CPU) {
        clock_t clk_start = clock();
        convert(h_img, h_img_gray, size);
        clock_t clk_end = clock();
        double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
        printf("[Grayscale conversion CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
        write_to_file(CPU_TIMING, "Grayscale", clk_elapsed, 0, 0);
    }
    else {
        cuda_convert(h_img, h_img_gray, width, height, N_STREAMS, GPU_TIMING);
    }

    free(img);
    free(h_img);

    if(WRITE)
        stbi_write_jpg("images/results/testGrayScale.jpg", width, height, 1, h_img_gray, 100);

    // Gamma correction on CPU/GPU
    if(CPU) {
        struct Histogram *hist = createHistogram();
        clock_t clk_start = clock();
        gamma_correction(hist, h_img_gray, size);
        clock_t clk_end = clock();
        double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
        free(hist);
        printf("[Gamma correction CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
        write_to_file(CPU_TIMING, "Gamma correction", clk_elapsed, 0, 0);
    }
    else {
        cuda_gamma_correction(h_img_gray, size, GPU_TIMING);
    }

    if(WRITE)
        stbi_write_jpg("images/results/testGammaCorrection.jpg", width, height, 1, h_img_gray, 100);

    unsigned char* gradientX = (unsigned char*)calloc(width*height, sizeof(unsigned char));
    unsigned char* gradientY = (unsigned char*)calloc(width*height, sizeof(unsigned char));

    // Gradients computation on CPU/GPU
    if(CPU) {
        clock_t clk_start = clock();
        convolutionHorizontal(h_img_gray, gradientX, height, width);
        convolutionVertical(h_img_gray, gradientY, height, width);
        clock_t clk_end = clock();
        double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
        printf("[Gradients computation CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
        write_to_file(CPU_TIMING, "Gradients", clk_elapsed, 0, 0);
    }
    else {
        cuda_compute_gradients(h_img_gray, gradientX, gradientY, width, height, GPU_TIMING);
    }

    free(h_img_gray);

    if(WRITE) {
        stbi_write_jpg("images/results/gradientX.jpg", width, height, 1, gradientX, 100);
        stbi_write_jpg("images/results/gradientY.jpg", width, height, 1, gradientY, 100);
    }

    unsigned char *magnitude = (unsigned char *)calloc(width*height, sizeof(unsigned char));
    unsigned char *direction = (unsigned char *)calloc(width*height, sizeof(unsigned char));

    // Magnitude and Direction computation on CPU/GPU
    if(CPU) {
        clock_t clk_start = clock();
        compute_magnitude(gradientX, gradientY, magnitude, width*height);
        compute_direction(gradientX, gradientY, direction, width*height);
        clock_t clk_end = clock();
        double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
        printf("[Magnitude & Direction CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
        write_to_file(CPU_TIMING, "Magnitude and Direction", clk_elapsed, 0, 0);
    }
    else {
        cuda_compute_mag_dir(gradientX, gradientY, magnitude, direction, width*height, GPU_TIMING);
    }

    free(gradientX);
    free(gradientY);

    if(WRITE) {
        stbi_write_jpg("images/results/magnitude.jpg", width, height, 1, magnitude, 100);
        stbi_write_jpg("images/results/direction.jpg", width, height, 1, direction, 100);
    }

    // HOG computation on CPU/GPU
    float *hog = NULL;

    if(CPU) {
        clock_t clk_start = clock();
        compute_hog(hog, magnitude, direction, width, height);
        clock_t clk_end = clock();
        double clk_elapsed = (double)(clk_end - clk_start)/CLOCKS_PER_SEC;
        printf("[HOG computation CPU] - Elapsed time: %.4f\n\n", clk_elapsed);
        write_to_file(CPU_TIMING, "HOG computation", clk_elapsed, 0, 1);
    }
    else {
        cuda_compute_hog(hog, magnitude, direction, width, height, GPU_TIMING);
    }

    printf("\n\n [DONE] \n\n");
}
