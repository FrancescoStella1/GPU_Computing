#include "../gradient.h"
#include "../timing.c"

#define TILE_WIDTH   (BLOCKDIM + MASK_SIZE - 1)


__constant__ int sobelX[MASK_SIZE*MASK_SIZE*sizeof(int)];
__constant__ int sobelY[MASK_SIZE*MASK_SIZE*sizeof(int)];



__global__ void convolutions_gpu(unsigned char *input_image, unsigned char *img_grad_x, unsigned char *img_grad_v, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = MASK_RADIUS;
    int block_m_radius = BLOCKDIM - radius;

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
            img_shared[threadIdx.y][threadIdx.x + radius] = input_image[(y-radius) * width + x ];  
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



void cuda_compute_gradients(unsigned char *img_gray, unsigned char *img_grad_h, unsigned char *img_grad_v, int width, int height, char *log_file) {
    unsigned char *d_img_grad, *d_grad_h, *d_grad_v;

    size_t size = width*height;

    const int h_sobelX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int h_sobelY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    const size_t mask_dim = sizeof(int)*MASK_SIZE*MASK_SIZE;

    CHECK(cudaMallocHost((void **)&d_img_grad, size));
    CHECK(cudaMallocHost((void **)&d_grad_h, size));
    CHECK(cudaMallocHost((void **)&d_grad_v, size));
    if(d_img_grad == NULL || d_grad_h == NULL || d_grad_v == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Copy sobel operators to constant memory
    CHECK(cudaMemcpyToSymbol(sobelX, &h_sobelX, mask_dim));
    CHECK(cudaMemcpyToSymbol(sobelY, &h_sobelY, mask_dim));

    CHECK(cudaMemcpyAsync(d_img_grad, img_gray, size, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    dim3 block(BLOCKDIM, BLOCKDIM);
    dim3 grid((width + BLOCKDIM - 1)/BLOCKDIM, (height + BLOCKDIM - 1)/BLOCKDIM);
    
    // Kernel launch
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    float time;
    cudaEventRecord(start, 0);
    convolutions_gpu<<<grid, block>>>(d_img_grad, d_grad_h, d_grad_v, width, height);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    printf("GPU Elapsed time: %f sec\n\n", time/1000);
    write_to_file(log_file, "Gradients", time/1000, 1, 0);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\n--> Error: %s\n", cudaGetErrorString(err));
    }

    // D2H transfer
    CHECK(cudaMemcpyAsync(img_grad_h, d_grad_h, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(img_grad_v, d_grad_v, size, cudaMemcpyDeviceToHost));

    // Host-Device synchronization
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFreeHost(d_img_grad));
    CHECK(cudaFreeHost(d_grad_h));
    CHECK(cudaFreeHost(d_grad_v));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}
