#!/bin/bash

nvcc -G -g -O3 main.cu -o main -Xptxas -dlcm=ca -I/usr/lib/x86_64-linux-gnu -lavutil -lavformat -lavcodec -lswscale -lm -lz -w
nvprof --devices 0 --metrics gld_efficiency,warp_execution_efficiency ./main $1