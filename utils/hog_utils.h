#define PI 3.14
#define HOG_BLOCK_SIDE 8
#define NUM_BINS 9
#define DELTA_THETA 20

#ifndef STRUCT_HOG
#define STRUCT_HOG

struct Hog {
    float *bins;
};

#endif

/**
 * Computes magnitude of the gradients.
 * @param gradientX pointer to the horizontal gradient array.
 * @param gradientY pointer to the vertical gradient array.
 * @param magnitude pointer to the magnitude of gradients array.
 * @param size size of the input arrays.
 *
*/
void compute_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size);


/**
 * Computes direction of the gradients.
 *
 * @param gradientX pointer to the horizontal gradient array.
 * @param gradientY pointer to the vertical gradient array.
 * @param direction pointer to the direction of gradients array.
 * @param size size of the input arrays.
 *
*/
void compute_direction(unsigned char *gradientX, unsigned char *gradientY, signed char *direction, int size);


/**
 * Computes Histogram of Oriented Gradients (HOG).
 * 
 * @param magnitude pointer to the magnitude of gradients array.
 * @param direction pointer to the direction of gradients array.
 * @param width width of the input arrays.
 * @param height height of the input arrays.
 *
*/
void compute_hog(unsigned char *magnitude, unsigned char *direction, int width, int height);


void cuda_compute_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size);


void cuda_compute_direction(unsigned char *gradientX, unsigned char *gradientY, unsigned char *direction, int size);