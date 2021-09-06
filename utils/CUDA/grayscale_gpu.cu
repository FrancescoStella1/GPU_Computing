#include "../grayscale.h"
#include "../common.h"

#define BLOCKDIM_GRAYSCALE   64
#define TILE   192


__global__ void grayscale_gpu_old(unsigned char *img, unsigned char *img_gray, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    
    uint idx = i*3;
    unsigned char r, g, b;
    r = img[idx];
    g = img[idx+1];
    b = img[idx+2];
    
    // grayscale conversion
    img_gray[i] = ((0.299*r) + (0.587*g) + (0.114*b));     // from it.mathworks.com - rgb2gray
}


__global__ void grayscale_gpu(unsigned char *img, unsigned char *img_gray, const size_t size) {
    
    __shared__ unsigned char tile[TILE];

    uint x, idx;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x>=size)
        return;

    tile[threadIdx.x] = img[x];
    __syncthreads();

    idx = threadIdx.x*3;

    unsigned char r, g, b;
    r = tile[idx];
    g = tile[idx + 1];
    b = tile[idx + 2];

    img_gray[x] = ((0.299*r) + (0.587*g) + (0.114*b));

}


void cuda_convert(unsigned char *h_img, unsigned char *h_img_gray, const size_t size) {
    // Device memory allocation
    unsigned char *d_img;
    unsigned char *d_img_gray;

    CHECK(cudaMalloc((void **)&d_img, size*3));   // 3 channels
    CHECK(cudaMalloc((void **)&d_img_gray, size));  
    if(d_img == NULL || d_img_gray == NULL)   {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Data transfer H2D
    CHECK(cudaMemcpy(d_img, h_img, size*3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));

    // Kernel launch
    dim3 block;
    dim3 grid;
    block.x = BLOCKDIM_GRAYSCALE;
    grid.x = ((size+block.x-1)/block.x);
    grayscale_gpu<<< grid, block >>>(d_img, d_img_gray, size);
    CHECK(cudaDeviceSynchronize());

    // Data transfer H2D
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_img));
    CHECK(cudaFree(d_img_gray));
}