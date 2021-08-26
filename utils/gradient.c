#include "gradient.h"


void convolutionHorizontal(unsigned char *img_gray, unsigned char *img_gradient, int row, int column) {
	char sobel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  	//ruota i pixel dell'immagine
  	for (int i = 0; i < row; i++){
		for (int j = 0; j < column; j++){
			int sum = 0;
      		//ruota i valori della maschera
			/*
			for (int k = -1; k <= 1; k++){
				int index = (i * column + j) - 1 + k;
				// controllo di non essere fuori dall riga
				if (index >= 0 && index < row * column)
					sum += img_gray[index] * k;
			}
			*/
			//for(int k = 0; k <= MASK_SIZE; k++) {
			int index = ((i - MASK_RADIUS) * column + j - MASK_RADIUS);
			//printf("First index: %d\n", index);
			if(index > column && i <= (row - MASK_RADIUS - 1) && j <= (column - MASK_RADIUS - 1)) {
				sum += img_gray[index-column-1] * sobel[0][2];
				sum += img_gray[index-column] * sobel[0][1];
				sum += img_gray[index-column+1] * sobel[0][0];
				//printf("First row of sobel completed.\n");
				sum += img_gray[index-1] * sobel[1][2];
				sum += img_gray[index] * sobel[1][1];
				sum += img_gray[index+1] * sobel[1][0];
				//printf("Second row of sobel completed.\n");
				sum += img_gray[index+column-1] * sobel[2][2];
				sum += img_gray[index+column] * sobel[2][1];
				sum += img_gray[index+column+1] * sobel[2][0];
				//printf("Third row of sobel completed.\n");
			//	}
			//img_gradient[i * column + j] = abs(sum);
				img_gradient[index] = abs(sum);
			}
		}
	}
}


void convolutionVertical(unsigned char *img_gray, unsigned char *img_gradient, int row, int column) {
	char sobel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
	//ruota i pixel dell'immagine
	for (int i = 0; i < row; i++){
		for (int j = 0; j < column; j++){
			int sum = 0;
      		//ruota i valori della maschera
			/*
			for (int k = -1; k <= 1; k++){
				int index = (i * column + j) + (column * (k - 1));
				// controllo di non essere fuori dall riga
				if (index >= 0 && index < row * column)
					sum += img_gray[index] * k;
			}
			*/
			int index = ((i - MASK_RADIUS) * column + j - MASK_RADIUS);
				//printf("First index: %d\n", index);
				if(index > column && i <= (row - MASK_RADIUS - 1) && j <= (column - MASK_RADIUS - 1)) {
					sum += img_gray[index-column-1] * sobel[0][2];
					sum += img_gray[index-column] * sobel[0][1];
					sum += img_gray[index-column+1] * sobel[0][0];
					//printf("First row of sobel completed.\n");
					sum += img_gray[index-1] * sobel[1][2];
					sum += img_gray[index] * sobel[1][1];
					sum += img_gray[index+1] * sobel[1][0];
					//printf("Second row of sobel completed.\n");
					sum += img_gray[index+column-1] * sobel[2][2];
					sum += img_gray[index+column] * sobel[2][1];
					sum += img_gray[index+column+1] * sobel[2][0];
					//printf("Third row of sobel completed.\n");
				//	}
				//img_gradient[i * column + j] = abs(sum);
					img_gradient[index] = abs(sum);
				//img_gradient[i * column + j] = abs(sum);
				}
		}
	}
}
