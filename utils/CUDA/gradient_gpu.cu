#include "../gradient.h"

#define BLOCKDIM   32
#define TILE   (BLOCKDIM + MASK_SIZE - 1)


__constant__ signed char sobelX[MASK_SIZE*MASK_SIZE];
__constant__ signed char sobelY[MASK_SIZE*MASK_SIZE];


__global__ void convolutionHorizontal_gpu(unsigned char *img_grad_h, unsigned char *g_img_grad_h, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = MASK_RADIUS;
    int BmR = BLOCKDIM - radius;
    //int m = MASK_SIZE;

    __shared__ int img_shared[TILE][TILE];

    if ((threadIdx.y < radius) ) {
    
        // left corner
        if (threadIdx.x < radius && (x-radius) >= 0 && (y-radius) >= 0)
            img_shared[threadIdx.y][threadIdx.x] = img_grad_h[(y-radius) * width + x - radius];
    
        // right corner
        if (threadIdx.x >= BmR && (x+radius) < width && (y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + 2*radius] = img_grad_h[(y-radius) * width + x + radius];
        
        // edge
        if ((y-radius) >= 0) 
            img_shared[threadIdx.y][threadIdx.x + radius] = img_grad_h[(y-radius) * width + x ];  
    }

    // Copy in column-major order the tile bottom halo 
    if (threadIdx.y >= BmR) {
    
        // left corner
        if (threadIdx.x < radius && (x-radius) >= 0 && (y+radius) < height)
            img_shared[threadIdx.y + 2*radius][threadIdx.x] = img_grad_h[(y+radius) * width + x - radius];

        // right corner
        if (threadIdx.x >= BmR && (y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + 2*radius] = img_grad_h[(y+radius) * width + x + radius];
    
        // edge
        if ((y+radius) < height) 
            img_shared[threadIdx.y + 2*radius][threadIdx.x + radius] = img_grad_h[(y+radius) * width + x];  
    }

    // Copy in column-major order the left halo
    if (threadIdx.x < radius) 
        if ((x-radius) >= 0) 
            img_shared[threadIdx.y + radius][threadIdx.x] = img_grad_h[y * width + x - radius];  

    // Copy in column-major order the right halo
    if (threadIdx.x >= BmR) 
        if ((x+radius) < width) 
            img_shared[threadIdx.y + radius][threadIdx.x + 2*radius] = img_grad_h[y * width + x + radius];  
      

    // Copy the tile center
	img_shared[radius + threadIdx.y][radius + threadIdx.x] = img_grad_h[y * width + x];
	
    // END SHARED MEMORY LOADING
	__syncthreads();

    int sum = 0.0;
	for (int i = 0; i < MASK_SIZE; i++)
		for (int j = 0; j < MASK_SIZE; j++)
			sum += img_shared[threadIdx.y+i][threadIdx.x+j] * sobelX[i*MASK_SIZE + j];
	
    __syncthreads();
    // write in global memory
    g_img_grad_h[x*width+y] = sum;
}


__global__ void convolutionVertical_gpu(unsigned char *img_grad_v, unsigned char *g_img_grad_v, int width, int height) {

}


void cuda_compute_gradients(unsigned char *img_gray_h, unsigned char *img_gray_v, int width, int height) {
    unsigned char *d_img_gray_h, *d_img_gray_v;
    unsigned char *d_grad_h, *d_grad_v;
    size_t mask_dim = MASK_SIZE*MASK_SIZE;
    size_t size = width*height;

    signed char h_sobelX[MASK_SIZE*MASK_SIZE] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    signed char h_sobelY[MASK_SIZE*MASK_SIZE] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    CHECK(cudaMallocHost((void **)&d_img_gray_h, size));
    CHECK(cudaMallocHost((void **)&d_img_gray_v, size));
    CHECK(cudaMallocHost((void **)&d_grad_h, size));
    CHECK(cudaMallocHost((void **)&d_grad_v, size));
    if(d_img_gray_h == NULL || d_img_gray_v == NULL || d_grad_h == NULL || d_grad_v == NULL) {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    // Copy sobel operators to constant memory
    CHECK(cudaMemcpyToSymbol(sobelX, h_sobelX, mask_dim, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(sobelY, h_sobelY, mask_dim, cudaMemcpyHostToDevice));

    
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
