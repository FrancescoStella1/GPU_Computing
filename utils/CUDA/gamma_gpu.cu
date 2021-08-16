#include "../gamma.h"
#include "../common.h"

#define BLOCKDIM   32


__global__ void gamma_correction_gpu(struct Histogram *hist, unsigned char *img_gray, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
        return;

    //atomicAdd(&(hist->num[img_gray[i]/L]), 1);      // provide L
    
}


void cuda_gamma_correction(struct Histogram *hist, unsigned char *h_img_gray, const size_t size) {
    unsigned char *d_img_gray;
    size_t nBytes = sizeof(struct Histogram);
    struct Histogram *d_hist;

    // Device memory allocation
    CHECK(cudaMalloc((void **)&d_hist, nBytes));
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    if(d_img_gray == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Data transfer H2D
    CHECK(cudaMemcpy(d_hist, hist, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));

    // Kernel launch
    dim3 block;
    dim3 grid;
    block.x = BLOCKDIM;
    grid.x = ((size + block.x - 1)/block.x);

    gamma_correction_gpu<<< grid, block >>>(hist, d_img_gray, size);
    CHECK(cudaDeviceSynchronize());

    // Data transfer D2H
    CHECK(cudaMemcpy(hist, d_hist, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_img_gray));
}