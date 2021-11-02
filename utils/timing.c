#pragma once
#include <time.h>
#include <stdio.h>

void write_to_file(char *filename, char *operation, double elapsed, int device, int final) {
    FILE *fp = fopen(filename, "a");
    if(fp == NULL) {
        fprintf(stderr, "Error opening the file %s\n", filename);
        exit(-1);
    }

    // Positioning at the end of the file
    if(fseek(fp, 0L, SEEK_END) != 0) {
        fprintf(stderr, "Unable to correctly open the file %s\n", filename);
        exit(-1);
    }
    printf("%ld\n", ftell(fp));
    fprintf(fp, operation);
    fprintf(fp, "\n\t");
    char *elapsed_str = (char *)malloc(6*sizeof(char));
    sprintf(elapsed_str, "%.6f", elapsed);
    fprintf(fp, elapsed_str);
    fprintf(fp, "\n");
    
    if(final)
        fprintf(fp, "\n###\n\n");

    fclose(fp);
}