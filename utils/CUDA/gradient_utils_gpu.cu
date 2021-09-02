
#define BLOCKDIM   32


__global__ void magnitude_gpu(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size)
        return;
    magnitude[i] = sqrt(pow(gradientX[i], 2) + pow(gradientY[i], 2));

}

void cuda_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size) {

    unsigned char *d_gradientX;
    unsigned char *d_gradientY;
    unsigned char *d_magnitude;

    CHECK(cudaMalloc((void **)&d_gradientX, size));
    CHECK(cudaMalloc((void **)&d_gradientY, size));
    CHECK(cudaMalloc((void **)&d_magnitude, size));

    if(d_gradientX == NULL || d_gradientY == NULL || d_magnitude == NULL)   {
        printf("Unable to allocate memory on GPU.\n");
        exit(EXIT_FAILURE);
    }

    CHECK(cudaMemcpy(d_gradientX, gradientX, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gradientY, gradientY, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_magnitude, magnitude, size, cudaMemcpyHostToDevice));

    dim3 block;
    dim3 grid;
    block.x = BLOCKDIM;
    grid.x = ((size+block.x-1)/block.x);

    magnitude_gpu<<< grid, block >>>(d_gradientX, d_gradientY, d_magnitude, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(magnitude, d_magnitude, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_gradientX));
    CHECK(cudaFree(d_gradientY));
    CHECK(cudaFree(d_magnitude));
}
