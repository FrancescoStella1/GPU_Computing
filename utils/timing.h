#include <time.h>
#include <stdio.h>


/**
 * Saves the timing of the functions into a log file.
 * 
 * @param filename filename in which to save the timings.
 * @param operation indicative name of the function/operation associated to the elapsed time.
 * @param elapsed elapsed time for the considered function.
 * @param device device used for the computation (0 = CPU | 1 = GPU).
 * @param final indicates if it is the last function of the HOG algorithm (0 = not final).
 * 
*/
void write_to_file(char *filename, char *operation, double elapsed, int device, int final);