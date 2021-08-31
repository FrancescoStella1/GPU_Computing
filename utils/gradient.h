#pragma once
#include <stdio.h>
#include <math.h>

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
 * @param img_gray_h input grayscale image used to compute horizontal gradient.
 * @param img_gray_v input grayscale image used to compute vertical gradient.
 * @param width width of the input grayscale image.
 * @param height height of the input grayscale image.
 * 
*/
void cuda_compute_gradients(unsigned char *img_gray_h, unsigned char *img_gray_v, int width, int height);