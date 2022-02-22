#pragma once
#include <stdio.h>
#include <math.h>

#define CONV_BLOCK_SIDE   32
#define MASK_RADIUS   1
#define MASK_SIZE   (2 * MASK_RADIUS +1)


/**
 * Calculates the gradient with horizontal convolution of a grayscale image.
 *
 * @param img_gray input grayscale image.
 * @param img_gradient output.
 * @param n_rows.
 * @param n_columns.
 *
*/
void convolutionHorizontal(unsigned char *img_gray, unsigned char *img_gradient, int n_rows, int n_columns);


/**
 * Calculates the gradient with vertical convolution of a grayscale image.
 *
 * @param img_gray input grayscale image.
 * @param img_gradient output.
 * @param n_rows.
 * @param n_columns.
 *
*/
void convolutionVertical(unsigned char *img_gray, unsigned char *img_gradient, int n_rows, int n_columns);


/**
 * Calculates horizontal and vertical gradients of a grayscale image.
 * 
 * @param img_gray pointer to grayscale image used to compute gradients.
 * @param img_grad_h pointer to the image representing the horizontal gradient of the original one.
 * @param img_grad_v pointer to the image representing the vertical gradient of the original one.
 * @param width width of the input grayscale image.
 * @param height height of the input grayscale image.
 * @param num_streams number of streams to use.
 * @param log_file file in which to save timings of the kernels.
 * @param write_timing int indicating if write of the timing is to be performed (0 if no write).
 * 
*/
void cuda_compute_gradients(unsigned char *img_gray, unsigned char *img_grad_h, unsigned char *img_grad_v, int width, int height, int num_streams, 
                            char *log_file, int write_timing);