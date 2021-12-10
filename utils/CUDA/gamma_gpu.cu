#include <stdio.h>
#include "../gamma.h"
#include "../common.h"

#define BLOCKDIM   64


__global__ void create_hist_gpu(unsigned int *num, unsigned char *img_gray, unsigned int *max_intensity, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
        return;

    // Create array s_num in order to store *num in shared memory
    __shared__ unsigned int s_num[256/L];
    s_num[threadIdx.x] = 0;

    __shared__ unsigned int s_intensity;
    unsigned int laneIdx = threadIdx.x % 32;
    unsigned int warpIdx = threadIdx.x / 32;

    if(threadIdx.x == 0)
      s_intensity = 0;

    unsigned int intensity = (unsigned int)img_gray[i];
    __syncthreads();                                              // s_num must be set entirely to 0

    atomicAdd((unsigned int *)&s_num[(intensity/L)], 1);

    for(int srcLane=1; srcLane<32; srcLane++) {
      int srcLaneVal = __shfl(intensity, srcLane);
      if(laneIdx == 0 && srcLaneVal > intensity)                  // only threads 0 and 32
        intensity = srcLaneVal;
    }

    __syncthreads();

    atomicAdd((unsigned int *)&num[threadIdx.x], s_num[threadIdx.x]);

    if(laneIdx == 0) {
      atomicMax((unsigned int *)&s_intensity, intensity);
    }
    __syncthreads();

    // Now s_intensity has the maximum intensity related to the current block

    if(threadIdx.x == 0) {
      atomicMax((unsigned int *)max_intensity, s_intensity);
    }
}


__global__ void merge_streams_results(unsigned int *hist_num, unsigned int *hist_num_streams, unsigned int *max_intensity, unsigned int *max_intensity_streams, int num_streams) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=(256/L))
        return;

    for(int idx=0; idx<num_streams; idx++) {
      atomicAdd((unsigned int *)&hist_num[i], hist_num_streams[i + idx*blockDim.x]);
    }

    if(i==0) {
      for(int idx=0; idx<num_streams; idx++) {
        atomicMax((unsigned int *)max_intensity, max_intensity_streams[idx]);
      }
    }

}


__global__ void apply_gamma_gpu(unsigned char *img_gray, double gamma, double factor, const size_t size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=size)
      return;
    
    img_gray[i] = (unsigned char)(factor*pow(img_gray[i], 1/gamma));

}


void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size, int num_streams, char *log_file) {
    struct Histogram *hist = createHistogram();
    unsigned int *h_max_intensity = (unsigned int *)malloc(sizeof(unsigned int));
    *h_max_intensity = 0;
    size_t nBytes = (256/L)*sizeof(unsigned int);
    size_t nBytes_1 = sizeof(unsigned int);
    unsigned int *d_num;
    unsigned char *d_img_gray;
    unsigned int *d_max_intensity;
    double g = 0;
    double factor = 0;

    // Device memory allocation
    CHECK(cudaMalloc((void **)&d_num, nBytes));
    CHECK(cudaMalloc((void **)&d_img_gray, size));
    CHECK(cudaMalloc((void **)&d_max_intensity, sizeof(unsigned int)));
    if(d_num == NULL || d_img_gray == NULL || d_max_intensity == NULL) {
      printf("Unable to allocate memory on GPU.\n");
      exit(EXIT_FAILURE);
    }
    CHECK(cudaMemset(d_num, 0, 256/L));
    
    dim3 block(BLOCKDIM);
    dim3 grid((size + block.x - 1)/block.x);
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time, time2;

    if(num_streams>1) {
      while((size % num_streams) != 0)
        num_streams++;
        
      int stream_size = size/num_streams;
      grid.x = (stream_size + block.x - 1)/block.x;
      
      cudaStream_t streams[num_streams];
      for(int idx=0; idx<num_streams; idx++) {
          CHECK(cudaStreamCreateWithFlags(&streams[idx], cudaStreamNonBlocking));
      }

      int stream_idx = 0;
      size_t nBytes_streams = num_streams*nBytes_1;

      unsigned int *h_max_intensity_pnd;
      unsigned int *hist_num_pnd;
      unsigned char *h_img_gray_pnd;
      unsigned int *d_max_intensity_streams;
      unsigned int *d_num_streams;

      CHECK(cudaHostAlloc((void **)&h_max_intensity_pnd, nBytes_streams, cudaHostAllocDefault));
      CHECK(cudaHostAlloc((void **)&hist_num_pnd, (256/L)*nBytes_streams, cudaHostAllocDefault));
      CHECK(cudaHostAlloc((void **)&h_img_gray_pnd, size, cudaHostAllocDefault));
      CHECK(cudaMalloc((void **)&d_max_intensity_streams, nBytes_streams));
      CHECK(cudaMalloc((void **)&d_num_streams, (256/L)*nBytes_streams));
      
      CHECK(cudaMemcpy(h_img_gray_pnd, h_img_gray, size, cudaMemcpyHostToHost));
      CHECK(cudaMemset(d_max_intensity_streams, 0, nBytes_streams));
      CHECK(cudaMemset(d_num_streams, 0, (256/L)*nBytes_streams));

      CHECK(cudaEventRecord(start, 0));

      for(int idx=0; idx<num_streams; idx++) {
        stream_idx = idx * stream_size;
        CHECK(cudaMemcpyAsync(&d_img_gray[stream_idx], &h_img_gray_pnd[stream_idx], stream_size, cudaMemcpyHostToDevice, streams[idx]));
        create_hist_gpu<<<grid, block, 0, streams[idx]>>>(&d_num_streams[idx*(256/L)], &d_img_gray[stream_idx], &d_max_intensity_streams[idx], stream_size);
        CHECK(cudaMemcpyAsync(&hist_num_pnd[idx*(256/L)], &d_num_streams[idx*(256/L)], nBytes, cudaMemcpyDeviceToHost, streams[idx]));
        CHECK(cudaMemcpyAsync(&h_max_intensity_pnd[idx], &d_max_intensity_streams[idx], nBytes_1, cudaMemcpyDeviceToHost, streams[idx]));
      }
      CHECK(cudaDeviceSynchronize());
      
      grid.x = 1;

      merge_streams_results<<<grid, block, 0>>>(d_num, d_num_streams, d_max_intensity, d_max_intensity_streams, num_streams);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(hist->num, d_num, nBytes, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_max_intensity, d_max_intensity, nBytes_1, cudaMemcpyDeviceToHost));
      
      CHECK(cudaEventRecord(end, 0));
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&time, start, end);
      time /= 1000;
      printf("[Gamma create histogram] - GPU Elapsed time: %f sec\n\n", time);

      CHECK(cudaFreeHost(h_max_intensity_pnd));
      CHECK(cudaFreeHost(hist_num_pnd));
      CHECK(cudaFreeHost(h_img_gray_pnd));
      CHECK(cudaFree(d_max_intensity));
      CHECK(cudaFree(d_max_intensity_streams));
      CHECK(cudaFree(d_num));
      CHECK(cudaFree(d_num_streams));

      g = compute_gamma(hist->num, hist->cnum, size);
      factor = *h_max_intensity/pow(*h_max_intensity, 1/g);

      printf("Normalized gamma value: %f\n", g);
      printf("Factor: %f\n", factor);
      printf("Max intensity: %u\n", *h_max_intensity);
      
      // Run second kernel
      CHECK(cudaEventRecord(start, 0));
      apply_gamma_gpu<<< grid, block >>>(d_img_gray, g, factor, size);
      CHECK(cudaDeviceSynchronize());
      
      // Data transfer D2H
      CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));

      CHECK(cudaEventRecord(end, 0));
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&time2, start, end);
      time2 /= 1000;
      printf("[Gamma Correction] - GPU Elapsed time: %f sec\n\n", (time + time2));
    }

    else {
      CHECK(cudaEventRecord(start, 0));
      // Data transfer H2D
      CHECK(cudaMemcpy(d_img_gray, h_img_gray, size, cudaMemcpyHostToDevice));
      //CHECK(cudaMemcpy(d_max_intensity, h_max_intensity, nBytes_1, cudaMemcpyHostToDevice));
      CHECK(cudaMemset(d_max_intensity, 0, nBytes_1));
      
      create_hist_gpu<<< grid, block >>>(d_num, d_img_gray, d_max_intensity, size);
      CHECK(cudaDeviceSynchronize());

      // Data transfer D2H
      CHECK(cudaMemcpy(hist->num, d_num, nBytes, cudaMemcpyDeviceToHost));
      //CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_max_intensity, d_max_intensity, nBytes_1, cudaMemcpyDeviceToHost));

      CHECK(cudaEventRecord(end, 0));

      cudaEventSynchronize(end);
      cudaEventElapsedTime(&time, start, end);
      time /= 1000;
      printf("[Gamma create histogram] - GPU Elapsed time: %f sec\n\n", time);
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess) {
          printf("\n--> Error: %s\n", cudaGetErrorString(err));
      }

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
      CHECK(cudaEventRecord(start, 0));
      apply_gamma_gpu<<< grid, block >>>(d_img_gray, g, factor, size);
      CHECK(cudaDeviceSynchronize());
      
      // Data transfer D2H
      CHECK(cudaMemcpy(h_img_gray, d_img_gray, size, cudaMemcpyDeviceToHost));
      
      CHECK(cudaEventRecord(end, 0));
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&time2, start, end);
      time2 /= 1000;
      printf("[Gamma Correction] - GPU Elapsed time: %f sec\n\n", (time + time2));
    }
    
    //write_to_file(log_file, "Gamma Correction", (time + time2), 1, 0);              // Generates Buffer Overflow in colab

    // Free memory
    CHECK(cudaFree(d_img_gray));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    free(h_max_intensity);
    delHistogram(hist);    
}