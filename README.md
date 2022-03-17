# GPU Computing project - Histogram of Oriented Gradients (HOG)

Project for the GPU Computing course attended at the University of Milan (Unimi), consisting in an implementation for the HOG algorithm with the only goal to show some possible speed-ups achievable through the use of GPUs.
The code is written in CUDA C and contains both CPU/GPU implementations of the algorithm.


## Usage

### Normal execution
For normal execution:

-  change the directives in the *main* and/or in the other files if needed.
-  use *make* after changing the makefile for the specific hardware (if needed).
-  call *main* with one of the two options (-i for processing images, -v for videos) followed by the path to the file.

### Profiling
For profiling, there are 3 bash scripts useful to retrieve some metrics. Please set the **WRITE_IMAGES** and **WRITE_TIMING** directives to 0 before the profiling and change other directives as needed.

### Generate plots

There are two python scripts for generating plots for comparison purposes. Before using them, please set the **WRITE_TIMING** directive to 1 and run the algorithm using both CPU and GPU. Then run the python scripts with the proper filenames from which to read execution times.


## Some achieved results


|   | CPU  &nbsp;&nbsp; | GPU (1 stream) &nbsp;&nbsp; | GPU (2 streams)  &nbsp;&nbsp; | GPU (4 streams) &nbsp;&nbsp; |
|---|:-----:|:----------------:|:-----------------:|:----------------:|
| immagini  | 0.107  | 0.004  | 0.003  | 0.003 |
| speed-up immagini | 0% | 96.3% | 97.2% | 97.2% |
| video | 59.05 | 2.04 | 1.76 | 1.59 |
| speed-up video | 0% | 96.54% | 97.02% | 97.31% |