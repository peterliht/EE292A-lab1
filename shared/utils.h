#ifndef __UTILS_H_
#define __UTILS_H_

void write_weights_file(char *filename, float *weights, int num_weights);

// bool read_weights_file(char *filename, int *weights);  //original: float *weights
// bool read_weights_file(char *filename, signed char *weights);  // 8b
bool read_weights_file(char *filename, ap_int<4> int4_t weights);  // 4b

unsigned int convert_endian_4bytes(unsigned int input);
int parse_MNIST_images(const char* file, unsigned char** X);
int parse_MNIST_labels(const char* file, unsigned char** y);
void parse_arguments(int argc, char *argv[], int *task, float *alpha, int *iterations, int *n_items_limit);
double get_wall_time();
#endif
