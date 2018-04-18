#include <stdio.h>
#include <stdlib.h>
#include "../shared/defines.h"
#include "../shared/utils.h"

unsigned char *X = NULL; 
unsigned char *y = NULL;
float *W = NULL;

void cleanup();

float predict(unsigned char* X, float* W){
	float hypothesis = 0;
	int k;
	for(k = 0; k < FEATURE_COUNT; k++){
		hypothesis += W[k] * X[k];
	}
	return hypothesis;
}

void classify(const char* images_file, const char* labels_file){
	bool status = true;
	char weights_file[256];
	int n_correct = 0;
	int n_items = parse_MNIST_images(images_file, &X);
	int items_tested = 0;
	
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		status = false;
	}
	if (status && n_items != parse_MNIST_labels(labels_file, &y)){
		printf("ERROR: Number of labels does not match number of images\n");
		status = false;
	}
	
	if (status) W = (float*) malloc(NUM_DIGITS * FEATURE_COUNT * sizeof(float));
		
	// Read in the weights from the weights files
	for (int i = 0; i < NUM_DIGITS && status; i++){
		snprintf(weights_file, 256, "weights_%d", i);
		status = status && read_weights_file(weights_file, W+FEATURE_COUNT*i);
	}
	
	// Start measuring classification time
	double start = get_wall_time();
	
	// Predict the digits on the test set
	for (int i = 0; i < n_items && status; i++){
		float max_score = VERY_NEGATIVE_NUMBER;
		unsigned char current_digit_guess = 0;
		for (int j = 0; j < NUM_DIGITS && status; j++){
			float score = predict(X+i*FEATURE_COUNT, W+FEATURE_COUNT*j);
			if (score > max_score){
				current_digit_guess = j;
				max_score = score;
			}
		}
		if (current_digit_guess == y[i]) n_correct++;
		items_tested++;
	}
	
	// Stop measuring the filter time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
	printf("Predicted %d correct out of %d (Accuracy: %.2f%%)\n", n_correct, items_tested, (float)n_correct * 100 / (float)items_tested);
	
	return;
}

int main(int argc, char *argv[]) {
	
	classify("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

	cleanup();
	
	return 0;
}

void cleanup() {
	if (W) free(W);
	if (X) free(X);
	if (y) free(y);
}
