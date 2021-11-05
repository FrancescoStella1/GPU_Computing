#include <stdio.h>
#include "../gamma.h"
#include "../common.h"

#define BLOCKDIM   32


__global__ void old_create_hist_gpu(unsigned int *num, unsigned char *img_gray, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
        return;
        
    atomicAdd((unsigned int *)&num[(img_gray[i]/L)], 1);
}

__global__ void create_hist_gpu(unsigned int *num, unsigned char *img_gray, unsigned int *max_intensity, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
        return;

    // Create array s_num in order to store *num in shared memory
    __shared__ unsigned int s_num[256/L];
    s_num[threadIdx.x] = 0;
    if(threadIdx.x >= 16) {
      s_num[(threadIdx.x*2)] = 0;
      s_num[(threadIdx.x*2)+1] = 0;
    }

    __syncthreads();
    
    // Variable in shared memory to find block-level maximum intensity
    __shared__ unsigned int s_intensity;
    if(threadIdx.x==0)
      s_intensity = 0;
    
    __syncthreads();

    unsigned int intensity = (unsigned int)img_gray[i];
    atomicAdd((unsigned int *)&s_num[(intensity/L)], 1);
    __syncthreads();
    
    for(int idx=256/L; idx>0; idx--) {
        if(s_num[idx] > 0 and (intensity/L) < idx)                // if there are intensities greater than the intensity of this thread, break the loop
          break;
        if(s_num[idx] > 0 and (intensity/L) == idx) {             // otherwise, compute the maximum intensity in the block
          atomicMax((unsigned int *)&s_intensity, intensity);
          break;
        }
    }
    __syncthreads();

    atomicAdd((unsigned int *)&num[threadIdx.x], s_num[threadIdx.x]);
    if(threadIdx.x >= 16) {
      atomicAdd((unsigned int *)&num[(threadIdx.x*2)], s_num[(threadIdx.x*2)]);
      atomicAdd((unsigned int *)&num[(threadIdx.x*2)+1], s_num[(threadIdx.x*2)+1]);
    }
    __syncthreads();

    if(threadIdx.x == 0) {
      atomicMax((unsigned int *)max_intensity, s_intensity);
    }
    __syncthreads();
}


__global__ void apply_gamma_gpu(unsigned char *img_gray, double gamma, double factor, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
      return;
    
    img_gray[i] = (unsigned char)(factor*pow(img_gray[i], 1/gamma));

}


void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size, char *log_file) {
    struct Histogram *hist = createHistogram();
    for(int idx=0; idx < (256/L); idx++) {
        hist->num[idx] = 0;
        hist->cnum[idx] = 0;
    }
    unsigned int *h_max_intensity = (unsigned int *)malloc(sizeof(unsigned int));
    *h_max_intensity = 0;
    size_t nBytes = (256/L)*sizeof(unsigned int);
    size_t nBytes_1 = sizeof(unsigned int);
    unsigned char *d_img_gray;
    unsigned int *d_num;
    unsigned int *d_max_intensity;
    double g = 0;
    double factor = 0;

    // Device memory allocation
    CHECK(cudaMalloc((unsigned int **)&d_num, nBytes));
    CHECK(cudaMalloc((unsigned char **)&d_img_gray, size));
    CHECK(cudaMalloc((unsigned int **)&d_max_intensity, sizeof(unsigned int)));
    if(d_num == NULL || d_img_gray == NULL || d_max_intensity == NULL) {
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
    CHECK(cudaMemcpy(d_max_intensity, h_max_intensity, nBytes_1, cudaMemcpyHostToDevice));

    // Run kernel
    cudaEventRecord(start, 0);
    create_hist_gpu<<< grid, block >>>(d_num, d_img_gray, d_max_intensity, size);
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
    CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_max_intensity, d_max_intensity, nBytes_1, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // Free memory
    CHECK(cudaFree(d_num));
    CHECK(cudaFree(d_max_intensity));


    // Compute cumulative histogram and normalized gamma value on CPU
    g = compute_gamma(hist->num, hist->cnum, size);
    factor = *h_max_intensity/pow(*h_max_intensity, 1/g);
    printf("Normalized gamma value: %f\n", g);
    printf("Factor: %f\n", factor);
    printf("Max intensity: %u\n", *h_max_intensity);
    
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