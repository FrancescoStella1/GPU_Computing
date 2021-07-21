#include <stddef.h>

/**
 * Converts a 3-channel image into a grayscale one.
 * 
 * @param img pointer to the image to convert.
 * @param grayscale pointer to the output grayscale image.
 * @param size size of the image to convert.
*/
void convert(unsigned char *img, unsigned char *grayscale, const size_t size);