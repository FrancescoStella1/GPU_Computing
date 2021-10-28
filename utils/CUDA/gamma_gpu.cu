#include <stdio.h>
#include "../gamma.h"
#include "../common.h"

#define BLOCKDIM   32


__global__ void create_hist_gpu(unsigned int *num, unsigned char *img_gray, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
        return;
        
    atomicAdd((unsigned int *)&num[(img_gray[i]/L)], 1);
}


__global__ void apply_gamma_gpu(unsigned char *img_gray, double gamma, double factor, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
      return;
    
    img_gray[i] = (unsigned char)(factor*pow(img_gray[i], 1/gamma));

}


void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size) {
    struct Histogram *hist = createHistogram();
    unsigned char *max_intensity = (unsigned char *)calloc(1, sizeof(unsigned char));
    size_t nBytes = (256/L)*sizeof(unsigned int);
    unsigned char *d_img_gray;
    unsigned int *d_num;
    double g = 0;
    double factor = 0;

    // Device memory allocation
    CHECK(cudaMalloc((void **)&d_num, nBytes));
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    if(d_num == NULL || d_img_gray == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    
    dim3 block(BLOCKDIM);
    dim3 grid((size + block.x - 1)/block.x);
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

    // Data transfer H2D
    CHECK(cudaMemcpy(d_num, hist->num, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));

    // Run kernel
    cudaEventRecord(start, 0);
    create_hist_gpu<<< grid, block >>>(d_num, d_img_gray, size);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("GPU Elapsed time: %f sec\n\n", time/1000);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    // Data transfer D2H
    CHECK(cudaMemcpy(hist->num, d_num, nBytes, cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_num));


    // Compute cumulative histogram and normalized gamma value on CPU
    g = compute_gamma(hist->num, hist->cnum, size, max_intensity);
    printf("Maximum pixel intensity in the grayscale image: %u\n", *max_intensity);
    factor = *max_intensity/pow(*max_intensity, 1/g);
    printf("Normalized gamma value: %f\n", g);
    printf("Factor: %f\n", factor);
    printf("Max intensity: %u\n", *max_intensity);
    
    // Run second kernel
    cudaEventRecord(start, 0);
    apply_gamma_gpu<<< grid, block >>>(d_img_gray, g, factor, size);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("GPU Elapsed time: %f sec\n\n", time/1000);

    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    // Data transfer D2H
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_img_gray));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    
}