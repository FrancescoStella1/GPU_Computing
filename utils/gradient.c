#include "gradient.h"


void convolutionHorizontal(unsigned char *img_gray, unsigned char *img_gradient, int n_rows, int n_columns) {
	char sobel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  	for (int i = 0; i < n_rows; i++){
		for (int j = 0; j < n_columns; j++){
			int sum = 0;
			int index = ((i - MASK_RADIUS) * n_columns + j - MASK_RADIUS);
			//printf("First index: %d\n", index);
			if(index > n_columns && i <= (n_rows - MASK_RADIUS - 1) && j <= (n_columns - MASK_RADIUS - 1)) {
				sum += img_gray[index-n_columns-1] * sobel[0][2];
				sum += img_gray[index-n_columns] * sobel[0][1];
				sum += img_gray[index-n_columns+1] * sobel[0][0];
				//printf("First row of sobel completed.\n");
				sum += img_gray[index-1] * sobel[1][2];
				sum += img_gray[index] * sobel[1][1];
				sum += img_gray[index+1] * sobel[1][0];
				//printf("Second row of sobel completed.\n");
				sum += img_gray[index+n_columns-1] * sobel[2][2];
				sum += img_gray[index+n_columns] * sobel[2][1];
				sum += img_gray[index+n_columns+1] * sobel[2][0];
				//printf("Third row of sobel completed.\n");
				img_gradient[index] = abs(sum);
			}
		}
	}
}


void convolutionVertical(unsigned char *img_gray, unsigned char *img_gradient, int n_rows, int n_columns) {
	char sobel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
	for (int i = 0; i < n_rows; i++){
		for (int j = 0; j < n_columns; j++){
			int sum = 0;
			int index = ((i - MASK_RADIUS) * n_columns + j - MASK_RADIUS);
				//printf("First index: %d\n", index);
				if(index > n_columns && i <= (n_rows - MASK_RADIUS - 1) && j <= (n_columns - MASK_RADIUS - 1)) {
					sum += img_gray[index-n_columns-1] * sobel[0][2];
					sum += img_gray[index-n_columns] * sobel[0][1];
					sum += img_gray[index-n_columns+1] * sobel[0][0];
					//printf("First row of sobel completed.\n");
					sum += img_gray[index-1] * sobel[1][2];
					sum += img_gray[index] * sobel[1][1];
					sum += img_gray[index+1] * sobel[1][0];
					//printf("Second row of sobel completed.\n");
					sum += img_gray[index+n_columns-1] * sobel[2][2];
					sum += img_gray[index+n_columns] * sobel[2][1];
					sum += img_gray[index+n_columns+1] * sobel[2][0];
					//printf("Third row of sobel completed.\n");
					img_gradient[index] = abs(sum);
				}
		}
	}
}
