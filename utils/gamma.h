#include <limits.h>
#include <math.h>
#include <stdlib.h>

#define L   4



struct Histogram {
    unsigned short *num;
    long *cnum;
};


/**
 * Allocates memory for 256/L bins.
 * 
 * @returns a pointer to a struct Histogram.
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