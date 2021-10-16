main.o: main.cu
	nvcc main.cu -o main -I/usr/lib/x86_64-linux-gnu -lavutil -lavformat -lavcodec -lswscale -lm -lz -w