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


void cuda_compute_mag_dir(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, int size) {

    unsigned char *d_gradientX;
    unsigned char *d_gradientY;
    unsigned char *d_magnitude;
    unsigned char *d_direction;

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

    dim3 block;
    dim3 grid;
    block.x = BLOCKDIM;
    grid.x = ((size+block.x-1)/block.x);

    mag_dir_gpu<<< grid, block >>>(d_gradientX, d_gradientY, d_magnitude, d_direction, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpyAsync(magnitude, d_magnitude, size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpyAsync(direction, d_direction, size, cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFreeHost(d_gradientX));
    CHECK(cudaFreeHost(d_gradientY));
    CHECK(cudaFreeHost(d_magnitude));
    CHECK(cudaFreeHost(d_direction));
}
