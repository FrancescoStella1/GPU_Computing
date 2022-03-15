#include "../gradient.h"
#include "../timing.c"

#define TILE_WIDTH   (BDIMX + MASK_SIZE - 1)
#define TILE_HEIGHT   (BDIMY + MASK_SIZE - 1)


__constant__ int sobelX[MASK_SIZE*MASK_SIZE*sizeof(int)];
__constant__ int sobelY[MASK_SIZE*MASK_SIZE*sizeof(int)];

typedef unsigned char uint40[5];

__global__ void convolutions_gpu(unsigned char *input_image, unsigned char *img_grad_x, unsigned char *img_grad_v, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = MASK_RADIUS;
    //int block_m_radius = CONV_BLOCK_SIDE - radius;
    int bmRx = BDIMX - radius;
    int bmRy = BDIMY - radius;

    __shared__ int img_shared[TILE_HEIGHT][TILE_WIDTH];
    //__shared__ uint4 img_shared[TILE_HEIGHT*sizeof(uint4)][TILE_WIDTH*sizeof(uint4)];
    
    if (x >= width || y >= height)
        return;
    
    // Top side of the block
    if ((threadIdx.y < radius) ) {

        // top left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y-radius) >= 0)
            img_shared[threadIdx.y][threadIdx.x] = input_image[(y-radius) * width + x - radius];
    
        // top right corner of the block
        if (threadIdx.x >= bmRx && (x+radius) < width && (y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + 2*radius] = input_image[(y-radius) * width + x + radius];
        
        // top side of the block
        if ((y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + radius] = input_image[(y-radius) * width + x];  
    }

    // Bottom side of the block
    if (threadIdx.y >= bmRy) {
    
        // bottom left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y+radius) < height)
            img_shared[threadIdx.y + 2*radius][threadIdx.x] = input_image[(y+radius) * width + x - radius];

        // bottom right corner of the block
        if (threadIdx.x >= bmRx && (y+radius) < height) 
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
    if (threadIdx.x >= bmRx) {
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
    
    // write in global memory the absolute values
    img_grad_x[y*width + x] = sum_x >= 0 ? sum_x : -sum_x;
    img_grad_v[y*width + x] = sum_y >= 0 ? sum_y : -sum_y;

}



void cuda_compute_gradients(unsigned char *img_gray, unsigned char *img_grad_h, unsigned char *img_grad_v, int width, int height, int num_streams, 
                            char *log_file, int write_timing) {
    unsigned char *d_img_gray, *d_grad_h, *d_grad_v;

    size_t size = width*height;

    const int h_sobelX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int h_sobelY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const size_t mask_dim = sizeof(int)*MASK_SIZE*MASK_SIZE;

    CHECK(cudaMalloc((void **)&d_img_gray, size));
    CHECK(cudaMalloc((void **)&d_grad_h, size));
    CHECK(cudaMalloc((void **)&d_grad_v, size));
    if(d_img_gray == NULL || d_grad_h == NULL || d_grad_v == NULL){
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    dim3 block(BDIMX, BDIMY);
    dim3 grid((width + BDIMX - 1)/BDIMX, (height + BDIMY - 1)/BDIMY);

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;

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

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    time /= 1000;
    //printf("[Gradients] - GPU Elapsed time: %f sec\n\n", time);
    
    if(write_timing)
        write_to_file(log_file, "Gradients", time, 1, 0);
    
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
