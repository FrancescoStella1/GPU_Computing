#include "grayscale.h"

void convert(unsigned char *img, unsigned char *img_gray, const size_t size) {
    unsigned char *p = img;
    for(unsigned char *p_gray = img_gray; p_gray<(img_gray + size); p_gray++) {
        *p_gray = ((0.299 * *p) + (0.587 * *(p+1)) + (0.114 * *(p+2)));     // from it.mathworks.com - rgb2gray
        p += 3;
    }
}