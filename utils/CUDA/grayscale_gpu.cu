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



void cuda_convert(unsigned char *h_img, unsigned char *h_img_gray, int width, int height, int num_streams, char *log_file) {
    // Device memory allocation
    unsigned char *d_img;
    unsigned char *d_img_gray;
    const size_t size = width*height*sizeof(unsigned char);

    dim3 block(BLOCKDIM);
    dim3 grid((size+block.x-1)/block.x);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

    CHECK(cudaMalloc((void **)&d_img, size*3));   // 3 channels
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    if(d_img == NULL || d_img_gray == NULL)   {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    if(num_streams>1) {
        while((size % num_streams) != 0)
            num_streams++;
        size_t size_streams_rgb = 3*(size/num_streams)*sizeof(unsigned char);
        size_t size_streams_gray = (size/num_streams)*sizeof(unsigned char);
        grid.x = (size_streams_gray+block.x-1)/block.x;         // recompute grid size
        
        cudaStream_t streams[num_streams];
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamCreateWithFlags(&streams[idx], cudaStreamNonBlocking));
        }
        
        unsigned char *h_img_gray_pnd;
        int rgb_idx = 0;
        int gray_idx = 0;

        CHECK(cudaHostAlloc((void **)&h_img_gray_pnd, size, cudaHostAllocDefault));

        CHECK(cudaEventRecord(start, 0));
        for(int idx=0; idx<num_streams; idx++) {
            rgb_idx = idx*size_streams_rgb;
            gray_idx = idx*size_streams_gray;
            CHECK(cudaMemcpyAsync(&d_img[rgb_idx], &h_img[rgb_idx], size_streams_rgb, cudaMemcpyHostToDevice, streams[idx]));
            grayscale_gpu<<<grid, block, 0, streams[idx]>>>(&d_img[rgb_idx], &d_img_gray[gray_idx], size_streams_gray);
            CHECK(cudaMemcpyAsync(&h_img_gray_pnd[gray_idx], &d_img_gray[gray_idx], size_streams_gray, cudaMemcpyDeviceToHost, streams[idx]));
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_img_gray, h_img_gray_pnd, size, cudaMemcpyHostToHost));
        CHECK(cudaEventRecord(end, 0));
        
        // Free some memory
        CHECK(cudaFreeHost(h_img_gray_pnd));
        
        // Destroy streams
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamDestroy(streams[idx]));
        }
    }

    else {

        cudaEventRecord(start, 0);
        // Data transfer H2D
        CHECK(cudaMemcpy(d_img, h_img, size*3, cudaMemcpyHostToDevice));
        //CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));
        // Kernel launch
        grayscale_gpu<<< grid, block >>>(d_img, d_img_gray, size);
        CHECK(cudaDeviceSynchronize());

        // Data transfer D2H
        CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));
        cudaEventRecord(end, 0);
    }

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    time /= 1000;
    printf("[Grayscale] - GPU Elapsed time: %f sec\n\n", time);
    //write_to_file(log_file, "Grayscale", time, 1, 0);                     // Generates Buffer Overflow in colab

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    // Free memory
    CHECK(cudaFree(d_img));
    CHECK(cudaFree(d_img_gray));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}