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
    
    img_gray[i] = (unsigned char)(factor*pow((double)*img_gray, 1/gamma));
}


void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size) {
    struct Histogram *hist = createHistogram();
    size_t nBytes = (256/L)*sizeof(unsigned int);
    unsigned char *d_img_gray;
    unsigned int *d_num;
    unsigned char max_intensity = 0;
    double g = 0;
    double factor = 0;

    // Device memory allocation
    CHECK(cudaMalloc((void **)&d_num, nBytes));
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    if(d_num == NULL || d_img_gray == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Data transfer H2D
    CHECK(cudaMemcpy(d_num, hist->num, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));

    dim3 block;
    dim3 grid;
    block.x = BLOCKDIM;
    grid.x = ((size + block.x - 1)/block.x);
    /*
    printf(" --- [FIRST ITERATION] --- \n\n");
    for(int idx=0; idx<(256/L); idx++) {
        printf("Bin %d: %u\n", idx, hist->num[idx]);
    }
    */

    // Run first kernel
    double start = seconds();
    create_hist_gpu<<< grid, block >>>(d_num, d_img_gray, size);
    CHECK(cudaDeviceSynchronize());
    double stop = seconds();
    printf("GPU Elapsed time: %f sec\n\n", stop-start);
    
    // Data transfer D2H
    CHECK(cudaMemcpy(hist->num, d_num, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Compute max intensity
    for(int idx=(256/L)-1; idx>=0; idx--) {
        if((hist->num[idx])>0)
            max_intensity = hist->num[idx] + (L/2);
    }
    printf("Maximum pixel intensity in the grayscale image: %u\n", max_intensity);

    // Free memory
    CHECK(cudaFree(d_img_gray));
    CHECK(cudaFree(d_num));

    // Compute cumulative histogram and normalized gamma value on CPU
    g = compute_gamma(hist->num, hist->cnum, size);
    factor = max_intensity/pow(max_intensity, 1/g);
    printf("Normalized gamma value: %f\n", g);

    // Reallocate device memory
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    if(d_img_gray == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Transfer H2D
    CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));
    
    // Run second kernel
    start = seconds();
    apply_gamma_gpu<<< grid, block >>>(d_img_gray, g, factor, size);
    CHECK(cudaDeviceSynchronize());
    stop = seconds();
    printf("GPU Elapsed time: %f sec\n\n", stop-start);

    // Data transfer D2H
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_img_gray));
    
    /*
    printf(" --- [SECOND ITERATION] --- \n\n");
    for(int idx=0; idx<(256/L); idx++) {
        printf("Bin %d: %u\n", idx, hist->num[idx]);
    }
    */
}