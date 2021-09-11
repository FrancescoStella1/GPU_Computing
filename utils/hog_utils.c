#include "hog_utils.h"


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


void compute_hog(unsigned char *magnitude, unsigned char *direction, int width, int height) {
  struct Hog *hog = (struct Hog *)malloc(sizeof(struct Hog));
  int num_blocks = (width*height)/HOG_BLOCK_SIDE + 1;
  hog->bins = (float *)calloc(NUM_BINS*num_blocks, sizeof(float));

  int row = 0;
  int col = 0;
  int block_idx = 0;
  
  while(row < height) {
    while(col < width) {
      for(int i=row; i<(height/HOG_BLOCK_SIDE); i++) {
        for(int j=col; j<(width/HOG_BLOCK_SIDE); j++) {
          int lbin = (direction[i*width + j] - DELTA_THETA/2)/DELTA_THETA;
          int ubin = lbin + 1;
          if(ubin>NUM_BINS-1)
            ubin=0;

          int cbin = DELTA_THETA*(lbin + 0.5);
          hog->bins[block_idx*NUM_BINS + lbin] += (magnitude[i*width + col] * ((direction[i*width + col] - DELTA_THETA/2)/DELTA_THETA));  // value of the j-th bin
          hog->bins[block_idx*NUM_BINS + ubin] += (magnitude[i*width + col] * ((direction[i*width + col] - cbin)/DELTA_THETA));
        }
      }
      block_idx += 1;
      col += (width/HOG_BLOCK_SIDE);
    }
    block_idx += 1;
    row += (height/HOG_BLOCK_SIDE);
    col = 0;
  }
  printf("Last row, col and blockIdx: %d %d %d\n", row, col, block_idx);
  for(int i=0; i<9; i++) {
    printf("%d Bin value: %.2f\n", i+1, hog->bins[i]);
  }

}