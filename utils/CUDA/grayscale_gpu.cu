#include "../grayscale.h"
#include "../common.h"
#include "../timing.c"

#define BLOCKDIM   32


__global__ void grayscale_gpu(unsigned char *img, unsigned char *img_gray, const size_t size) {
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



void cuda_convert(unsigned char *h_img, unsigned char *h_img_gray, int width, int height, char *log_file) {
    // Device memory allocation
    unsigned char *d_img;
    unsigned char *d_img_gray;
    const size_t size = width*height;

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
    dim3 block(BLOCKDIM);
    dim3 grid((size+block.x-1)/block.x);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;
    cudaEventRecord(start, 0);
    grayscale_gpu<<< grid, block >>>(d_img, d_img_gray, size);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("[Grayscale] - GPU Elapsed time: %f sec\n\n", time/1000);
    write_to_file(log_file, "Grayscale", time/1000, 1, 0);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    // Data transfer H2D
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

    // Free memory
    CHECK(cudaFree(d_img));
    CHECK(cudaFree(d_img_gray));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}