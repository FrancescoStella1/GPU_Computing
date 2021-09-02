#include "../gradient.h"

#define BLOCKDIM   32
#define TILE_WIDTH   (BLOCKDIM + MASK_SIZE - 1)


__constant__ int sobelX[MASK_SIZE*MASK_SIZE];
__constant__ int sobelY[MASK_SIZE*MASK_SIZE];


__global__ void convolutionHorizontal_gpu(unsigned char *in_img_grad, unsigned char *out_img_grad, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    int radius = MASK_RADIUS;
    int block_m_radius = BLOCKDIM - radius;

    __shared__ int img_shared[TILE_WIDTH][TILE_WIDTH];

    // Top side of the block
    if ((threadIdx.y < radius) ) {
    
        // top left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y-radius) >= 0)
            img_shared[threadIdx.y][threadIdx.x] = in_img_grad[(y-radius) * width + x - radius];
    
        // top right corner of the block
        if (threadIdx.x >= block_m_radius && (x+radius) < width && (y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + 2*radius] = in_img_grad[(y-radius) * width + x + radius];
        
        // top side of the block
        if ((y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + radius] = in_img_grad[(y-radius) * width + x ];  
    }

    // Bottom side of the block
    if (threadIdx.y >= block_m_radius) {
    
        // bottom left corner of the block
        if (threadIdx.x < radius && (x-radius) >= 0 && (y+radius) < height)
            img_shared[threadIdx.y + 2*radius][threadIdx.x] = in_img_grad[(y+radius) * width + x - radius];

        // bottom right corner of the block
        if (threadIdx.x >= block_m_radius && (y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + 2*radius] = in_img_grad[(y+radius) * width + x + radius];
    
        // bottom side of the block
        if ((y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + radius] = in_img_grad[(y+radius) * width + x];  
    }

    // Left side of the block
    if (threadIdx.x < radius) {
        if ((x-radius) >= 0) {
            img_shared[threadIdx.y + radius][threadIdx.x] = in_img_grad[y * width + x - radius];  
        }
    }

    // Right side of the block
    if (threadIdx.x >= block_m_radius) {
        if ((x+radius) < width) {
            img_shared[threadIdx.y + radius][threadIdx.x + 2*radius] = in_img_grad[y * width + x + radius];  
        }
    }
      

    // center of the block
	img_shared[radius + threadIdx.y][radius + threadIdx.x] = in_img_grad[y * width + x];
	
    // END SHARED MEMORY LOADING
	__syncthreads();

    int sum = 0;
	for (int i = 0; i < MASK_SIZE; i++) {
		for (int j = 0; j < MASK_SIZE; j++) {
			//sum += img_shared[threadIdx.y + i][threadIdx.x + j] * sobelX[i*MASK_SIZE + j];
            sum += img_shared[threadIdx.y][threadIdx.x] * sobelX[i*MASK_SIZE + j];
        }
    }
	
    __syncthreads();
    // write in global memory
    //in_img_grad[x*width+y] = sum;
    in_img_grad[idx] = sum;
}


__global__ void convolutionVertical_gpu(unsigned char *img_grad_v, unsigned char *g_img_grad_v, int width, int height) {

}


void cuda_compute_gradients(unsigned char *img_gray_h, unsigned char *img_gray_v, int width, int height) {
    unsigned char *d_img_gray_h, *d_img_gray_v;
    unsigned char *d_grad_h, *d_grad_v;
    size_t mask_dim = MASK_SIZE*MASK_SIZE;
    size_t size = width*height;

    const int h_sobelX[mask_dim] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int h_sobelY[mask_dim] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    CHECK(cudaMallocHost((void **)&d_img_gray_h, size));
    CHECK(cudaMallocHost((void **)&d_img_gray_v, size));
    CHECK(cudaMallocHost((void **)&d_grad_h, size));
    CHECK(cudaMallocHost((void **)&d_grad_v, size));
    if(d_img_gray_h == NULL || d_img_gray_v == NULL || d_grad_h == NULL || d_grad_v == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Copy sobel operators to constant memory
    CHECK(cudaMemcpyToSymbol(sobelX, &h_sobelX, mask_dim, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(sobelY, &h_sobelY, mask_dim, cudaMemcpyHostToDevice));

    
    // Non-blocking stream creation
    cudaStream_t stream1, stream2;
    CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    /*
    // Asynchronous data transfer H2D and kernel execution
    CHECK(cudaMemcpyAsync(d_grad_h, img_gray_h, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(d_grad_v, img_gray_v, size, cudaMemcpyHostToDevice));
    */
    CHECK(cudaMemcpyAsync(d_img_gray_h, img_gray_h, size, cudaMemcpyHostToDevice, stream1));
    CHECK(cudaMemcpyAsync(d_grad_h, img_gray_h, size, cudaMemcpyHostToDevice, stream1));

    dim3 block(BLOCKDIM, BLOCKDIM);
    dim3 grid((width + BLOCKDIM - 1)/BLOCKDIM, (height + BLOCKDIM - 1)/BLOCKDIM);

    convolutionHorizontal_gpu<<<grid, block, 0, stream2>>>(d_img_gray_h, d_grad_h, width, height);
    CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    // D2H transfer
    CHECK(cudaMemcpy(img_gray_h, d_img_gray_h, width*height, cudaMemcpyDeviceToHost));
    //convolutionVertical_gpu<<<grid, block>>>();

    CHECK(cudaFreeHost(d_img_gray_h));
    CHECK(cudaFreeHost(d_img_gray_v));
    CHECK(cudaFreeHost(d_grad_h));
    CHECK(cudaFreeHost(d_grad_v));
}
