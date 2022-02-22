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
 * @return normalized gamma value.
*/
double compute_gamma(unsigned int *num, long *cnum, const size_t size);


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
 * @param num_streams number of streams to use.
 * @param size size of the grayscale image.
*/
void gamma_correction(struct Histogram *hist, unsigned char *img_gray, int num_streams, const size_t size);


/**
 * Applies Gamma correction algorithm on GPU.
 * 
 * @param h_img_gray host pointer to the grayscale image.
 * @param size size of the grayscale image.
 * @param log_file file in which to save timings of the kernels.
 * @param write_timing int indicating if write of the timing is to be performed (0 if no write).
*/
void cuda_gamma_correction(unsigned char *h_img_gray, const size_t size, char *log_file, int write_timing);