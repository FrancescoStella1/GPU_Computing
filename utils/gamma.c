#include "gamma.h"


struct Histogram *createHistogram() {
    struct Histogram *hist = (struct Histogram *)malloc(sizeof(struct Histogram));
    hist->num = (unsigned int *)calloc((256/L), sizeof(unsigned int));
    hist->cnum = (long *)calloc((256/L), sizeof(long));
    return hist;
}


double compute_gamma(unsigned int *num, long *cnum, const size_t size) {
    // First cumulative bin equal to the first original bin
    cnum[0] = num[0];
    for(int i=1; i<(256/L); i++) {
        cnum[i] = (num[i] + cnum[i-1]);
        //printf("Bin %d: %ld\n", i+1, cnum[i]);
    }

    // Compute cumulative histogram
    long lower_threshold = size*0.05;
    long upper_threshold = size*0.95;
    long min, median, max;
    min = 0;
    median=0;
    max=0;
    long median_th = (upper_threshold-lower_threshold)/2;
    //printf("Lower: %ld\nUpper: %ld\n", lower_threshold, upper_threshold);
    for(int i=0; i<(256/L); i++) {
        long tmp = cnum[i];
        if(tmp>=lower_threshold) {
            if(min==0)
                min=i;
            if(tmp>=upper_threshold && max==0)
                max=i;
            if(tmp>=median_th && median==0) {
                median=i;
            }
        }
    }
    printf("CMin: %lu\nCMax: %lu\nMedian: %lu\n", min, max, median);

    // Compute gamma value
    printf("gamam is log of: %f\n", (double)(median-min)/(max-median));
    double g = log2((double)(median-min)/(max-median));
    printf("gamma value: %f\n", g);

    // Normalize g
    if(g<0.8)
        g=0.8;
    else if(g>1.2)
        g=1.2;
    printf("Normalized gamma value: %f\n", g);

    return g;
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
    float max_intensity=0;
    for(unsigned char *p=img_gray; p<(img_gray + size); p++) {
        if(*p>max_intensity)
            max_intensity=*p;
        hist->num[*p/L] += 1;
    }

    double g = compute_gamma(hist->num, hist->cnum, size);
    double factor = max_intensity/pow(max_intensity, 1/g);

    // Apply gamma correction
    for(unsigned char *p=img_gray; p<(img_gray + size); p++) {
        *p = (unsigned char)(factor*(pow(*p, 1/g)));
    }
}