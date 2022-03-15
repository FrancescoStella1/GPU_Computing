#define PI   3.14
#define MAGDIR_BLOCK_SIZE   64
#define HOG_BLOCK_WIDTH   32
#define HOG_BLOCK_HEIGHT   32
#define HOG_BLOCK_SIDE   32
#define NUM_BINS   9
#define DELTA_THETA   20


/**
 * @brief Creates the histograms for computing HOG.
 * 
 * @param width width of the image.
 * @param height height of the image.
 * @return size_t size of the memory to be allocated for storing HOG.
 */
size_t allocate_histograms(int width, int height);


/**
 * @brief Computes magnitude of the gradients.
 * 
 * @param gradientX pointer to the horizontal gradient array.
 * @param gradientY pointer to the vertical gradient array.
 * @param magnitude pointer to the magnitude of gradients.
 * @param size size of the input gradients.
 *
*/
void compute_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size);


/**
 * @brief Computes direction of the gradients.
 *
 * @param gradientX pointer to the horizontal gradient array.
 * @param gradientY pointer to the vertical gradient array.
 * @param direction pointer to the direction of gradients.
 * @param size size of the input gradients.
 *
*/
void compute_direction(unsigned char *gradientX, unsigned char *gradientY, unsigned char *direction, int size);


/**
 * @brief Computes Histogram of Oriented Gradients (HOG).
 * 
 * @param hog pointer to the histograms (one for each 8x8 block in the image).
 * @param magnitude pointer to the magnitude of gradients array.
 * @param direction pointer to the direction of gradients array.
 * @param width width of the input arrays.
 * @param height height of the input arrays.
 *
*/
void compute_hog(float *hog, unsigned char *magnitude, unsigned char *direction, int width, int height);


/**
 * @brief Computes magnitude and direction of the gradients.
 * 
 * @param gradientX pointer to the horizontal gradient array.
 * @param gradientY pointer to the vertical gradient array.
 * @param magnitude pointer to the magnitude of the gradients.
 * @param direction pointer to the direction of the gradients.
 * @param width width of the gradient image.
 * @param height height of the gradient image.
 * @param num_streams number of streams to use.
 * @param log_file file in which to save timings of the kernels.
 * @param write_timing int indicating if write of the timing is to be performed (0 if no write).
*/
void cuda_compute_mag_dir(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, unsigned char *direction, int width, int height, 
                          int num_streams, char *log_file, int write_timing);


/**
 * @brief Compute HOG from magnitude and direction.
 * 
 * @param hog pointer to the histograms (one for each 8x8 block in the image).
 * @param magnitude pointer to the magnitude vector.
 * @param direction pointer to the direction vector.
 * @param width width of the image.
 * @param height height of the image.
 * @param num_streams number of streams to use.
 * @param log_file file in which to save timings of the kernels.
 * @param write_timing int indicating if write of the timing is to be performed (0 if no write).
*/
void cuda_compute_hog(float *hog, unsigned char *magnitude, unsigned char *direction, int width, int height, int num_streams, char *log_file, int write_timing);