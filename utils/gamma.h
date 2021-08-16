#include <limits.h>
#include <math.h>
#include <stdlib.h>

#define L   4


#ifndef STRUCT_HIST
#define STRUCT_HIST

struct Histogram {
    unsigned short *num;
    long *cnum;
};

#endif


/**
 * Allocates memory for 256/L bins.
 * 
 * @return a pointer to a struct Histogram.
*/
struct Histogram *createHistogram();


/**
 * Frees memory previously allocated for bins.
 * 
 * @param hist pointer to a struct Histogram.
*/
void delHistogram(struct Histogram *hist);


/**
 * Applies Gamma correction algorithm.
 * 
 * @param hist pointer to a struct Histogram.
 * @param img_gray pointer to the grayscale image.
 * @param size size of the grayscale image.
*/
void gamma_correction(struct Histogram *hist, unsigned char *img_gray, const size_t size);


/**
 * Applies Gamma correction algorithm on GPU.
 * 
 * @param hist pointer to a struct Histogram.
 * @param h_img_gray host pointer to the grayscale image.
 * @param size size of the grayscale image.
*/
void cuda_gamma_correction(struct Histogram *hist, unsigned char *h_img_gray, const size_t size);