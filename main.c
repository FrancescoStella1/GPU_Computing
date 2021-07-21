#define STB_IMAGE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include "./stb_image/stb_image.h"
#include "./stb_image/stb_image_write.h"


int main (int argc, char **argv) {

    int width, height, channels;
        unsigned char *load;
        if (argc > 1)
            load = stbi_load(argv[1], &width, &height, &channels, 0);
        else
            load = stbi_load("images/calciatore.jpg", &width, &height, &channels, 0);
        if (load == NULL){
            printf("Error loading the image... \n");
            exit(EXIT_FAILURE);
        }
        printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n\n", width, height, channels);

    
}
