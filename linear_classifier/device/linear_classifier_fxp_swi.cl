#define ARRAY_DIM 784 // 28*28
#define NUM_DIGITS 10

__kernel void linear_classifier(global const unsigned char * restrict images, 
								global int * restrict weights,
								global unsigned char * restrict guesses,
								const int num_images)
{
	unsigned char guess = 0;
	int score[10] = {0};
	
	int local_weights[ARRAY_DIM*NUM_DIGITS];
	
	for (int i = 0; i < ARRAY_DIM*NUM_DIGITS; i++){
		local_weights[i] = weights[i];
	}
	
	for (int n = 0; n < num_images; n++){
		
		guess = 0;
		
		#pragma unroll
		for (int i = 0; i < 10; i++)
			score[i] = 0;
		
		#pragma unroll 8
		for (int x = 0; x < ARRAY_DIM; x++){
			#pragma unroll
			for (int i = 0; i < 10; i++){
				score[i] += images[n*ARRAY_DIM + x]*local_weights[i*ARRAY_DIM+x];
			}
		}
	
		// Determine highest score
		#pragma unroll
		for (int i = 1; i < 10; i++){
			if (score[i] > score[guess]) guess = i;
		}
		guesses[n] = guess;
	}
}

