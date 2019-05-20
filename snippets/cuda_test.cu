#include <stdio.h>
#include <math.h>
// Kernel function to add the elements of two arrays
#include <curand_kernel.h>
//#include <pycuda-complex.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <ctime>

#define X_MIN -1.5f
#define X_MAX 1.5f
#define Y_MIN -3.2f
#define Y_MAX 2.0f
#define X_DIM 1440
#define Y_DIM 2560
#define ITERS 20
#define N (1<<16)

//typedef pycuda::complex<float> cmplx;

__device__ curandState_t* states[N];

__global__
void init_kernel(int seed) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) {
		curandState_t* s = new curandState_t;
		if (s != 0) {
			curand_init(seed, idx, 0, s);
		}

		states[idx] = s;
	} else {
		printf("forbidden memory access %%d/%%d\\n", idx, N);
	}
}

__device__
void to_pixel(float &px, float &py, int &ix, int &iy) {
	px -= X_MIN;
	py -= Y_MIN;
	px /= X_MAX - X_MIN;
	py /= Y_MAX - Y_MIN;
	px *= X_DIM;
	py *= Y_DIM;
	ix = __float2int_rd(px);
	iy = __float2int_rd(py);
}

__device__
void write_pixel(int idx, float px, float py, int ix, int iy,
	float4 *z, unsigned int *canvas) {
	px = z[idx].y;
	py = z[idx].x;
	to_pixel(px, py, ix, iy);
	if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
		canvas[iy*X_DIM + ix] += 1;
	}
}


__device__
void generate_random_complex(float real, float imag, int idx,
	float4 *z, float *dists, unsigned int *counts) {

	real *= X_MAX-X_MIN+3;
	real += X_MIN-2;
	imag *= Y_MAX-Y_MIN+0;
	imag += Y_MIN-0;

	z[idx].x = real;
	z[idx].y = imag;
	z[idx].z = real;
	z[idx].w = imag;
	dists[idx] = 0;
	counts[idx] = 0;
}

__global__
void buddha_kernel(unsigned int *counts, float4 *z,
	float *dists, unsigned int *canvas) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, ix, iy;
	float real, imag;//, temp0, temp1;

	if (idx < N) {
		curandState_t s = *states[idx];
		#pragma unroll 4
		for(i = 0; i < 10000; i++) {

			real = curand_uniform(&s);
			imag = curand_uniform(&s);
			generate_random_complex(real, imag, idx, z, dists, counts);
			to_pixel(real, imag, ix, iy);

			while (counts[idx] < ITERS & dists[idx] < 25) {
				counts[idx]++;
				real = z[idx].x*z[idx].x - z[idx].y*z[idx].y + z[idx].z;
				imag = 2*z[idx].x*z[idx].y + z[idx].w;
				z[idx].x = real;
				z[idx].y = imag;
				dists[idx] = z[idx].x*z[idx].x + z[idx].y*z[idx].y;
			}

			if (dists[idx] > 25) {
				z[idx].x = 0;
				z[idx].y = 0;
				for (j = 0; j < counts[idx]+1; j++) {
					real = z[idx].x*z[idx].x - z[idx].y*z[idx].y + z[idx].z;
					imag = 2*z[idx].x*z[idx].y + z[idx].w;
					z[idx].x = real;
					z[idx].y = imag;
					write_pixel(idx, real, imag, ix, iy, z, canvas);
				}
			}
		}
		*states[idx] = s;
	} else {
		printf("forbidden memory access %%d/%%d\\n", idx, N);
	}
}

int writeFile (unsigned int *canvas, double elapsed) {
	std::ofstream myfile;
	myfile.open ("example.txt");
	double sum = 0;
	// myfile << "Writing this to a file.\n";
	for (int i = 0; i < Y_DIM; i++) {
		for (int j = 0; j < X_DIM-1; j++) {
			myfile << canvas[i*X_DIM + j] << ",";
			sum += canvas[i*X_DIM + j];
		}
		myfile << canvas[(i+1)*X_DIM-1];
		sum += canvas[(i+1)*X_DIM - 1];
		if (i < Y_DIM-1) {
			myfile << "\n";
		}
	}
	printf("\tTotal iterations: %.2e\n", sum);
	printf("\tIterations per second: %.2e\n", (sum/elapsed));

	myfile.close();
	return 0;
}




int main(void) {
	unsigned int *counts, *canvas;
	float4 *z;
	float *dists;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&counts, 	N*sizeof(unsigned int));
	cudaMallocManaged(&canvas, 	X_DIM*Y_DIM*sizeof(unsigned int));
	cudaMallocManaged(&z,		N*sizeof(float4));
	cudaMallocManaged(&dists,	N*sizeof(float));

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	init_kernel<<<numBlocks, blockSize>>>(123);
	std::clock_t begin = std::clock();
	buddha_kernel<<<numBlocks, blockSize>>>(counts, z, dists, canvas);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("\tTotal time: %.2fs\n", elapsed_secs);

	writeFile(canvas, elapsed_secs);
	// Free memory
	cudaFree(counts);
	cudaFree(canvas);
	cudaFree(z);
	cudaFree(dists);
	printf("Buddhabrot generated!\n");
	return 0;
}