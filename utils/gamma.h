#pragma once
#include <limits.h>
#include <math.h>
#include <stdlib.h>


#define L   4


#ifndef STRUCT_HIST
#define STRUCT_HIST


struct Histogram {
    unsigned int *num;
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
 * Computes gamma by creating a cumulative histogram and retrieving min, median and max values.
 * 
 * @param num pointer to the histogram.
 * @param cnum pointer to the cumulative histogram.
 * @param size size of the grayscale image.
 * @param max_intensity pointer to the maximum intensity value in the grayscale image, default to NULL.
 * @return normalized gamma value.
*/
double compute_gamma(unsigned int *num, long *cnum, const size_t size, unsigned char *max_intensity);


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
 * @param h_img_gray host pointer to the grayscale image.
 * @param size size of the grayscale image.
*/
void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size);