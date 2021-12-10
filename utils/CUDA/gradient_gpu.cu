#include "../gradient.h"
#include "../timing.c"

#define TILE_WIDTH   (CONV_BLOCK_SIDE + MASK_SIZE - 1)


__constant__ int sobelX[MASK_SIZE*MASK_SIZE*sizeof(int)];
__constant__ int sobelY[MASK_SIZE*MASK_SIZE*sizeof(int)];



__global__ void convolutions_gpu(unsigned char *input_image, unsigned char *img_grad_x, unsigned char *img_grad_v, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = MASK_RADIUS;
    int block_m_radius = CONV_BLOCK_SIDE - radius;

    __shared__ unsigned char img_shared[TILE_WIDTH][TILE_WIDTH];

    // Top side of the block
    if ((threadIdx.y < radius) ) {

        // top left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y-radius) >= 0)
            img_shared[threadIdx.y][threadIdx.x] = input_image[(y-radius) * width + x - radius];
    
        // top right corner of the block
        if (threadIdx.x >= block_m_radius && (x+radius) < width && (y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + 2*radius] = input_image[(y-radius) * width + x + radius];
        
        // top side of the block
        if ((y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + radius] = input_image[(y-radius) * width + x];  
    }

    // Bottom side of the block
    if (threadIdx.y >= block_m_radius) {
    
        // bottom left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y+radius) < height)
            img_shared[threadIdx.y + 2*radius][threadIdx.x] = input_image[(y+radius) * width + x - radius];

        // bottom right corner of the block
        if (threadIdx.x >= block_m_radius && (y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + 2*radius] = input_image[(y+radius) * width + x + radius];
    
        // bottom side of the block
        if ((y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + radius] = input_image[(y+radius) * width + x];  
    }

    // Left side of the block
    if (threadIdx.x < radius) {
        if ((x-radius) >= 0) {
            img_shared[threadIdx.y + radius][threadIdx.x] = input_image[y * width + x - radius];  
        }
    }

    // Right side of the block
    if (threadIdx.x >= block_m_radius) {
        if ((x+radius) < width) {
            img_shared[threadIdx.y + radius][threadIdx.x + 2*radius] = input_image[y * width + x + radius];  
        }
    }
      
    // center of the block
	img_shared[radius + threadIdx.y][radius + threadIdx.x] = input_image[y * width + x];
	
    // END SHARED MEMORY LOADING
	__syncthreads();
    
    int sum_x = 0;
    int sum_y = 0;
	for (int i = 0; i < MASK_SIZE; i++) {
		for (int j = 0; j < MASK_SIZE; j++) {
            sum_x += img_shared[threadIdx.y + i][threadIdx.x + j] * sobelX[i*MASK_SIZE + j];
            sum_y += img_shared[threadIdx.y + i][threadIdx.x + j] * sobelY[i*MASK_SIZE + j];
        }
    }
	
    __syncthreads();
    
    // write in global memory
    img_grad_x[y*width + x] = abs(sum_x);
    img_grad_v[y*width + x] = abs(sum_y);
}



void cuda_compute_gradients(unsigned char *img_gray, unsigned char *img_grad_h, unsigned char *img_grad_v, int width, int height, int num_streams, char *log_file) {
    unsigned char *d_img_gray, *d_grad_h, *d_grad_v;

    size_t size = width*height;

    const int h_sobelX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int h_sobelY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const size_t mask_dim = sizeof(int)*MASK_SIZE*MASK_SIZE;

    CHECK(cudaMalloc((void **)&d_img_gray, size));
    CHECK(cudaMalloc((void **)&d_grad_h, size));
    CHECK(cudaMalloc((void **)&d_grad_v, size));
    if(d_img_gray == NULL || d_grad_h == NULL || d_grad_v == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    dim3 block(CONV_BLOCK_SIDE, CONV_BLOCK_SIDE);
    dim3 grid((width + CONV_BLOCK_SIDE - 1)/CONV_BLOCK_SIDE, (height + CONV_BLOCK_SIDE - 1)/CONV_BLOCK_SIDE);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

    if(num_streams>1) {
        while((size % num_streams) != 0)
            num_streams++;
        
        int stream_size = size/num_streams;
        grid.x = (stream_size + block.x - 1)/block.x;
        
        cudaStream_t streams[num_streams];
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamCreateWithFlags(&streams[idx], cudaStreamNonBlocking));
        }

        // Pinned memory allocation
        unsigned char *img_gray_pnd, *img_grad_h_pnd, *img_grad_v_pnd;
        int stream_idx = 0;
        CHECK(cudaHostAlloc((void **)&img_gray_pnd, size, cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void **)&img_grad_h_pnd, size, cudaHostAllocDefault));
        CHECK(cudaHostAlloc((void **)&img_grad_v_pnd, size, cudaHostAllocDefault));

        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpy(img_gray_pnd, img_gray, size, cudaMemcpyHostToHost));
        CHECK(cudaMemcpyToSymbol(sobelX, &h_sobelX, mask_dim));
        CHECK(cudaMemcpyToSymbol(sobelY, &h_sobelY, mask_dim));

        for(int idx=0; idx<num_streams; idx++) {
            stream_idx = idx * stream_size;
            CHECK(cudaMemcpyAsync(&d_img_gray[stream_idx], &img_gray_pnd[stream_idx], stream_size, cudaMemcpyHostToDevice, streams[idx]));
            convolutions_gpu<<<grid, block, 0, streams[idx]>>>(&d_img_gray[stream_idx], &d_grad_h[stream_idx], &d_grad_v[stream_idx], width, height);
            CHECK(cudaMemcpyAsync(&img_grad_h_pnd[stream_idx], &d_grad_h[stream_idx], stream_size, cudaMemcpyDeviceToHost, streams[idx]));
            CHECK(cudaMemcpyAsync(&img_grad_v_pnd[stream_idx], &d_grad_v[stream_idx], stream_size, cudaMemcpyDeviceToHost, streams[idx]));
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(img_grad_h, img_grad_h_pnd, size, cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(img_grad_v, img_grad_v_pnd, size, cudaMemcpyHostToHost));
        CHECK(cudaEventRecord(end, 0));

        // Free some memory
        CHECK(cudaFreeHost(img_gray_pnd));
        CHECK(cudaFreeHost(img_grad_h_pnd));
        CHECK(cudaFreeHost(img_grad_v_pnd));

        // Destroy streams
        for(int idx=0; idx<num_streams; idx++) {
            CHECK(cudaStreamDestroy(streams[idx]));
        }
    }

    else {
        // Data transfer H2D
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpyToSymbol(sobelX, &h_sobelX, mask_dim));
        CHECK(cudaMemcpyToSymbol(sobelY, &h_sobelY, mask_dim));
        CHECK(cudaMemcpy(d_img_gray, img_gray, size, cudaMemcpyHostToDevice));
        
        convolutions_gpu<<<grid, block>>>(d_img_gray, d_grad_h, d_grad_v, width, height);
        CHECK(cudaDeviceSynchronize());
        
        // D2H transfer
        CHECK(cudaMemcpy(img_grad_h, d_grad_h, size, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(img_grad_v, d_grad_v, size, cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(end, 0));
    }

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    time /= 1000;
    printf("[Gradients] - GPU Elapsed time: %f sec\n\n", time);
    //write_to_file(log_file, "Gradients", time, 1, 0);                     // Generates Buffer Overflow in colab
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    CHECK(cudaFree(d_img_gray));
    CHECK(cudaFree(d_grad_h));
    CHECK(cudaFree(d_grad_v));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}
