#!/bin/bash

nvcc -G -g -O3 main.cu -o main -Xptxas -dlcm=ca -I/usr/lib/x86_64-linux-gnu -lavutil -lavformat -lavcodec -lswscale -lm -lz -w
nvprof --devices 0 --metrics gld_transactions_per_request,gst_transactions_per_request --unified-memory-profiling off ./main $1