#include "../hog_utils.h"


__global__ void mag_dir_gpu(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, int size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    
    float grad_x = gradientX[i];
    float grad_y = gradientY[i];

    float mag = sqrtf(powf(grad_x, 2) + powf(grad_y, 2));
    float atang = atan2f(grad_y, grad_x) * (180/PI);

    magnitude[i] = (unsigned char)mag;
    direction[i] = (unsigned char)atang;

}


__global__ void hog_gpu(float *bins, unsigned char *magnitude, unsigned char *direction, int width, int height) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= width || j >= height)
        return;

    int lbin = direction[i*width + j]/DELTA_THETA;
    int ubin = lbin + 1;
    if(ubin>=NUM_BINS)
      ubin = 0;

    int cbin = (lbin + 0.5);

    float l_value = magnitude[i*width + j] * ((direction[i*width + j] - DELTA_THETA/2)/DELTA_THETA);  // value of the j-th bin
    float u_value = magnitude[i*width + j] * ((direction[i*width + j] - cbin)/DELTA_THETA);

    int blocks_per_row = (width + HOG_BLOCK_SIDE - 1)/HOG_BLOCK_SIDE;
    int block_idx = blockIdx.y * blocks_per_row + blockIdx.x;
    atomicAdd(&bins[block_idx*NUM_BINS + lbin], l_value);
    atomicAdd(&bins[block_idx*NUM_BINS + ubin], u_value);
}


void cuda_compute_mag_dir(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, int dim) {

    unsigned char *d_gradientX;
    unsigned char *d_gradientY;
    unsigned char *d_magnitude;
    unsigned char *d_direction;
    size_t size = dim;

    CHECK(cudaMallocHost((void **)&d_gradientX, size));
    CHECK(cudaMallocHost((void **)&d_gradientY, size));
    CHECK(cudaMallocHost((void **)&d_magnitude, size));
    CHECK(cudaMallocHost((void **)&d_direction, size));

    if(d_gradientX == NULL || d_gradientY == NULL || d_magnitude == NULL || d_direction == NULL)   {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    CHECK(cudaMemcpyAsync(d_gradientX, gradientX, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_gradientY, gradientY, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_magnitude, magnitude, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_direction, direction, size, cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    dim3 block(HOG_BLOCK_SIDE);
    dim3 grid((size+block.x-1)/block.x);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;
    
    cudaEventRecord(start, 0);
    mag_dir_gpu<<< grid, block >>>(d_gradientX, d_gradientY, d_magnitude, d_direction, dim);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("GPU Elapsed time: %f sec\n\n", time/1000);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    CHECK(cudaMemcpyAsync(magnitude, d_magnitude, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(direction, d_direction, size, cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFreeHost(d_gradientX));
    CHECK(cudaFreeHost(d_gradientY));
    CHECK(cudaFreeHost(d_magnitude));
    CHECK(cudaFreeHost(d_direction));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}


void cuda_compute_hog(unsigned char *magnitude, unsigned char *direction, int width, int height) {
    unsigned char *d_magnitude, *d_direction;
    float *d_bins;
    size_t size = width*height;
    int num_blocks = (width*height)/HOG_BLOCK_SIDE + 1;
    size_t nBytes = NUM_BINS*num_blocks*sizeof(float);

    struct Hog *hog = (struct Hog *)malloc(sizeof(struct Hog));
    hog->bins = (float *)calloc(NUM_BINS*num_blocks, sizeof(float));
    
    CHECK(cudaMallocHost((void **)&d_magnitude, size));
    CHECK(cudaMallocHost((void **)&d_direction, size));
    CHECK(cudaMallocHost((void **)&d_bins, nBytes));
    if(d_magnitude == NULL || d_direction == NULL || d_bins == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // To do: implement streams
    CHECK(cudaMemcpyAsync(d_magnitude, magnitude, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_direction, direction, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_bins, hog->bins, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    dim3 block(HOG_BLOCK_SIDE, HOG_BLOCK_SIDE);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);


    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;
    
    cudaEventRecord(start, 0);
    hog_gpu<<< grid, block >>>(d_bins, d_magnitude, d_direction, width, height);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("GPU Elapsed time: %f sec\n\n", time/1000);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    CHECK(cudaMemcpyAsync(hog->bins, d_bins, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFreeHost(d_magnitude));
    CHECK(cudaFreeHost(d_direction));
    CHECK(cudaFreeHost(d_bins));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}