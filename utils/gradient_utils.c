#define PI 3.14


//calcola la magnitude del gradiente
void compute_magnitude(unsigned char *gradientX, unsigned char *gradientY, unsigned char *magnitude, int size){
  for (int i = 0; i < size; i++) {
    magnitude[i] = sqrt(pow(gradientX[i], 2) + pow(gradientY[i], 2));
  }
}

//calcola la direzione del gradiente
void compute_direction(unsigned char *gradientX, unsigned char * gradientY, unsigned char *direction, int size) {
	for (int i = 0; i < size; i++) {
    direction[i] = atan2(gradientY[i], gradientX[i]) * (180 / PI);
  }
}
