#include "../hog_utils.h"
#include "../timing.c"


__global__ void mag_dir_gpu(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, size_t size) {
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

    unsigned char mag = magnitude[j*width + i];
    unsigned char dir = direction[j*width + i];
    int lbin = dir/DELTA_THETA;
    int ubin = lbin + 1;
    if(ubin>=NUM_BINS)
      ubin = 0;

    int cbin = (lbin + 0.5);

    float l_value =  mag * ((dir - (DELTA_THETA/2))/DELTA_THETA);
    float u_value = mag * ((dir - cbin)/DELTA_THETA);

    int blocks_per_row = (width + HOG_BLOCK_WIDTH - 1)/HOG_BLOCK_WIDTH;
    int block_idx = blockIdx.y * blocks_per_row + blockIdx.x;

    atomicAdd(&bins[block_idx*NUM_BINS + lbin], l_value);
    atomicAdd(&bins[block_idx*NUM_BINS + ubin], u_value);
}


void cuda_compute_mag_dir(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, int width, int height, 
                          int num_streams, char *log_file, int write_timing) {

    unsigned char *d_gradientX;
    unsigned char *d_gradientY;
    unsigned char *d_magnitude;
    unsigned char *d_direction;
    size_t size = width*height*sizeof(unsigned char);

    CHECK(cudaMalloc((void **)&d_gradientX, size));
    CHECK(cudaMalloc((void **)&d_gradientY, size));
    CHECK(cudaMalloc((void **)&d_magnitude, size));
    CHECK(cudaMalloc((void **)&d_direction, size));
    if(d_gradientX == NULL || d_gradientY == NULL || d_magnitude == NULL || d_direction == NULL)   {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    dim3 block(MAGDIR_BLOCK_SIZE);
    dim3 grid((size+block.x-1)/block.x);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

    if(num_streams>1) {
        while((size % num_streams) != 0)
            num_streams++;
        size_t stream_size = size/num_streams;

        grid.x = (stream_size + block.x - 1)/block.x;

        cudaStream_t streams[num_streams];
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamCreateWithFlags(&streams[idx], cudaStreamNonBlocking));
        }

        int stream_idx = 0;

        unsigned char *gradientX_pnd;
        unsigned char *gradientY_pnd;
        unsigned char *magnitude_pnd;
        unsigned char *direction_pnd;

        CHECK(cudaHostAlloc((void **)&gradientX_pnd, size, cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void **)&gradientY_pnd, size, cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void **)&magnitude_pnd, size, cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void **)&direction_pnd, size, cudaHostAllocDefault));

        CHECK(cudaEventRecord(start, 0));

        CHECK(cudaMemcpyAsync(gradientX_pnd, gradientX, size, cudaMemcpyHostToHost));
        CHECK(cudaMemcpyAsync(gradientY_pnd, gradientY, size, cudaMemcpyHostToHost));
        CHECK(cudaDeviceSynchronize());

        for(int idx=0; idx<num_streams; idx++) {
            stream_idx = idx * stream_size;
            CHECK(cudaMemcpyAsync(&d_gradientX[stream_idx], &gradientX_pnd[stream_idx], stream_size, cudaMemcpyHostToDevice, streams[idx]));
            CHECK(cudaMemcpyAsync(&d_gradientY[stream_idx], &gradientY_pnd[stream_idx], stream_size, cudaMemcpyHostToDevice, streams[idx]));
            mag_dir_gpu<<<grid, block, 0, streams[idx]>>>(&d_gradientX[stream_idx], &d_gradientY[stream_idx], &d_magnitude[stream_idx], &d_direction[stream_idx], stream_size);
            CHECK(cudaMemcpyAsync(&magnitude_pnd[stream_idx], &d_magnitude[stream_idx], stream_size, cudaMemcpyDeviceToHost, streams[idx]));
            CHECK(cudaMemcpyAsync(&direction_pnd[stream_idx], &d_direction[stream_idx], stream_size, cudaMemcpyDeviceToHost, streams[idx]));
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpyAsync(magnitude, magnitude_pnd, size, cudaMemcpyHostToHost));
        CHECK(cudaMemcpyAsync(direction, direction_pnd, size, cudaMemcpyHostToHost));

        CHECK(cudaDeviceSynchronize());

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        time /= 1000;
        //printf("[Magnitude & Direction] - GPU Elapsed time: %f sec\n\n", time);

        // Free some memory
        CHECK(cudaFreeHost(gradientX_pnd));
        CHECK(cudaFreeHost(gradientY_pnd));
        CHECK(cudaFreeHost(magnitude_pnd));
        CHECK(cudaFreeHost(direction_pnd));

        // Destroy non-null streams
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamDestroy(streams[idx]));
      }
    }

    else {
        CHECK(cudaMemcpy(d_gradientX, gradientX, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_gradientY, gradientY, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_magnitude, magnitude, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_direction, direction, size, cudaMemcpyHostToDevice));

        cudaEventRecord(start, 0);
        mag_dir_gpu<<< grid, block >>>(d_gradientX, d_gradientY, d_magnitude, d_direction, size);
        CHECK(cudaDeviceSynchronize());
        
        CHECK(cudaMemcpy(magnitude, d_magnitude, size, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(direction, d_direction, size, cudaMemcpyDeviceToHost));
        
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        time /= 1000;
        //printf("[Magnitude & Direction] - GPU Elapsed time: %f sec\n\n", time);

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("\n--> Error: %s\n", cudaGetErrorString(err));
        }
    }
    
    if(write_timing)
        write_to_file(log_file, "Magnitude and Direction", time, 1, 0);

    CHECK(cudaFree(d_gradientX));
    CHECK(cudaFree(d_gradientY));
    CHECK(cudaFree(d_magnitude));
    CHECK(cudaFree(d_direction));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}


void cuda_compute_hog(float *hog, unsigned char *magnitude, unsigned char *direction, int width, int height, int num_streams, char *log_file, 
                      int write_timing) {
                          
    unsigned char *d_magnitude;
    unsigned char *d_direction;
    float *d_bins;
    size_t size = width*height;
    int blocks_per_row = (width + HOG_BLOCK_WIDTH - 1)/HOG_BLOCK_WIDTH;
    int blocks_per_col = (height + HOG_BLOCK_HEIGHT - 1)/HOG_BLOCK_HEIGHT;
    int num_blocks = blocks_per_row * blocks_per_col; //(size + HOG_BLOCK_SIDE - 1)/HOG_BLOCK_SIDE;
    size_t nBytes = NUM_BINS*num_blocks*sizeof(float);
    // size_t hog_size = allocate_histograms(width, height);
    // hog = (float *)malloc(nBytes);
    
    CHECK(cudaMalloc((void **)&d_magnitude, size));
    CHECK(cudaMalloc((void **)&d_direction, size));
    CHECK(cudaMalloc((void **)&d_bins, nBytes));
    if(d_magnitude == NULL || d_direction == NULL || d_bins == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }
    //CHECK(cudaMemset(d_bins, 0, nBytes));

    dim3 block(HOG_BLOCK_WIDTH, HOG_BLOCK_HEIGHT);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

    CHECK(cudaMemcpy(d_magnitude, magnitude, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_direction, direction, size, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    cudaEventRecord(start, 0);
    hog_gpu<<< grid, block >>>(d_bins, d_magnitude, d_direction, width, height);
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(hog, d_bins, nBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    time /= 1000;
    //printf("[HOG Computation] - GPU Elapsed time: %f sec\n\n", time);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    //for(int i=0; i<10; i++) {
        //printf("HOG %d: %.2f\n", i, hog[i]);
    //}

    if(write_timing)
        write_to_file(log_file, "HOG computation", time, 1, 1);

    CHECK(cudaFree(d_magnitude));
    CHECK(cudaFree(d_direction));
    CHECK(cudaFree(d_bins));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}