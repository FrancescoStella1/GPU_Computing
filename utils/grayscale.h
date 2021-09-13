#pragma once
#include <stddef.h>

/**
 * Converts a 3-channel image into a grayscale one.
 * 
 * @param img pointer to the image to convert.
 * @param img_gray pointer to the grayscale image.
 * @param size size of the converted image.
*/
void convert(unsigned char *img, unsigned char *img_gray, const size_t size);


/**
 * Converts on GPU a 3-channel image into a grayscale one.
 * 
 * @param h_img host pointer to the image to convert.
 * @param h_img_gray host pointer to the grayscale image.
 * @param width width of the image.
 * @param height height of the image.
*/
void cuda_convert(unsigned char *h_img, unsigned char *h_img_gray, int width, int height);