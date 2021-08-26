#pragma once
#include <stdio.h>
#include <math.h>

#define MASK_RADIUS   1
#define MASK_SIZE   (2 * MASK_RADIUS +1)


/**
 * Calculates the gradient with horizontal convolution of a grayscale image
 *
 * @param img_gray input grayscale image.
 * @param img_gradient output.
 * @param row.
 * @param column.
 *
*/
void convolutionHorizontal(unsigned char *img_gray, unsigned char *img_gradient, int row, int column);


/**
 * Calculates the gradient with vertical convolution of a grayscale image
 *
 * @param img_gray input grayscale image.
 * @param img_gradient output.
 * @param row.
 * @param column.
 *
*/
void convolutionVertical(unsigned char *img_gray, unsigned char *img_gradient, int row, int column);
