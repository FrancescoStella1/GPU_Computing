#include "hog_utils.h"


size_t allocate_histograms_old(int num_blocks) {
  size_t num = NUM_BINS * num_blocks;
  // float *hog = (float *)calloc(NUM_BINS*num_blocks, sizeof(float));
  return num;
}

size_t allocate_histograms(int width, int height) {
  int num_blocks = (width*height + HOG_BLOCK_SIDE - 1)/HOG_BLOCK_SIDE;
  size_t num = NUM_BINS * num_blocks;
  return num;
}

void delete_histograms(float *hog) {
  if(hog != NULL)
    free(hog);
}


//calcola la magnitude del gradiente
void compute_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size){
  for (int i = 0; i < size; i++) {
    magnitude[i] = sqrtf(powf(gradientX[i], 2) + powf(gradientY[i], 2));
  }
}

//calcola la direzione del gradiente
void compute_direction(unsigned char *gradientX, unsigned char * gradientY, unsigned char *direction, int size) {
	for (int i = 0; i < size; i++) {
    direction[i] = atan2f(gradientY[i], gradientX[i]) * (180 / PI);
  }
}


void compute_hog(float *hog, unsigned char *magnitude, unsigned char *direction, int width, int height) {
  int num_blocks = (width*height + HOG_BLOCK_SIDE - 1)/HOG_BLOCK_SIDE;
  // size_t size = allocate_histograms(width, height);
  // hog = (float *)calloc(size, sizeof(float));
  //size_t hog_size = NUM_BINS * num_blocks * sizeof(float);
  //hog = (float *)malloc(hog_size);

  int row = 0;
  int col = 0;
  int block_idx = 0;
  
  while(row < height) {
    while(col < width) {
      for(int i = row; i < row + HOG_BLOCK_SIDE; i++) {
        for(int j = col; j < col + HOG_BLOCK_SIDE; j++) {
          int lbin = (direction[i*width + j] - DELTA_THETA/2)/DELTA_THETA; //direction[i*width + j]/DELTA_THETA;
          int ubin = lbin + 1;
          if(ubin>=NUM_BINS)
            ubin = 0;

          int cbin = (lbin + 0.5);

          float l_value = magnitude[i*width + j] * ((direction[i*width + col] - DELTA_THETA/2)/DELTA_THETA);  // value of the j-th bin
          float u_value = magnitude[i*width + j] * ((direction[i*width + col] - cbin)/DELTA_THETA);
          hog[block_idx*NUM_BINS + lbin] += l_value;
          hog[block_idx*NUM_BINS + ubin] += u_value;
        }
      }
      col += HOG_BLOCK_SIDE;
    }
    block_idx += 1;
    row += HOG_BLOCK_SIDE;
    col = 0;
  }
  
  //for(int idx=0; idx<NUM_BINS; idx++) {
    //printf("HOG - bin %d: %.2f\n", idx, hog[idx]);
  //}
  
}