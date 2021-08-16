#include "gamma.h"


struct Histogram *createHistogram() {
    struct Histogram *hist = (struct Histogram *)malloc(sizeof(struct Histogram));
    hist->num = (unsigned short *)calloc((256/L), sizeof(unsigned short)); //malloc((256/L)*sizeof(unsigned short));
    hist->cnum = (long *)calloc((256/L), sizeof(long));
    return hist;
}


void delHistogram(struct Histogram *hist) {
    if(hist != NULL) {
        free(hist->num);
        free(hist->cnum);
        free(hist);
    }
}


void gamma_correction(struct Histogram *hist, unsigned char *img_gray, const size_t size) {
    // Populate histogram
    for(unsigned char *p=img_gray; p<(img_gray + size); p++) {
        hist->num[*p/L] += 1;
    }

    printf("\n\n---------\n\n");
    // First cumulative bin equal to the first original bin
    hist->cnum[0] = hist->num[0];
    for(int i=1; i<(256/L); i++) {
        hist->cnum[i] = (hist->num[i] + hist->cnum[i-1]);
        printf("Bin %d: %ld\n", i, hist->cnum[i]);
    }

    // Compute cumulative histogram
    long lower_threshold = size*0.05;
    long upper_threshold = size*0.95;
    float min, median, max;
    min = 0;
    max = 0;
    median = 0;
    printf("Lower: %ld\nUpper: %ld\n", lower_threshold, upper_threshold);
    for(int i=0; i<(256/L); i++) {
        long tmp = hist->cnum[i];
        if(tmp>lower_threshold) {
            if(min==0)
                min=i*L;
            if(tmp>=upper_threshold && max==0)
                max=i*L;
            if(tmp>=(upper_threshold+lower_threshold)/2 && median==0) {
                float m = (hist->cnum[i+1] - hist->cnum[i])/4;
                median=((upper_threshold+lower_threshold)/2)/m;
            }
        }
    }
    printf("CMin: %.4f\nCMax: %.4f\nMedian: %.4f\n", min, max, median);

    // Compute gamma value
    float g = log((median-min)/(max-median));
    //float g = log((float)(hist->cnum[median] - hist->cnum[min]) / (hist->cnum[max] - hist->cnum[median])); //(median-min)/(max-median));
    printf("gamma value: %f\n", g);

    // Normalize g
    if(g<0.8)
        g=0.8;
    else if(g>1.2)
        g=1.2;
    printf("Normalized gamma value: %f\n", g);

    // Apply gamma correction
    for(unsigned char *p=img_gray; p<(img_gray + size); p++)
        *p *= (pow((*p/max), 1/g));
}