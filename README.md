# CUDA-optimized Buddhabrot

This repo was created to host my optimized implemetation of a Buddhabrot generator using PyCuda.

## Requirements

* CUDA 9.0
* CUDA 9.0 Samples (used for `helper_math.h`)
* PyCuda built for CUDA 9.0
* `numpy`
* `scipy`
* `matplotlib` (only for generating graphs during benchmarking)


## Usage

Running the code is as simple as doing `python3 buddha_kernel.py`.

## Examples

### 20 iteration count

The grainy image was generated in 20ms, the fine image was generated in 1840ms.

<p align="center">
	<img src="/images/example_grainy_20.png" width="30%" />
	<img src="/images/example_fine_20.png" width="30%" />
</p>

### 100 iteration count

The grainy image was generated in 10ms, the fine image was generated in 2490ms.

<p align="center">
	<img src="/images/example_grainy_100.png" width="30%" />
	<img src="/images/example_fine_100.png" width="30%" />
</p>

## Performance

This is some example output from my laptop (GTX 1050) and my desktop (GTX 1080Ti). In both tests, settings are adjusted to make the execution time about 30 seconds for the actual buddha calculation. However, the important distinction between the two tests is the total number of iterations, or the number of iterations per second. For the GTX 1050, a speed of about 750 Msamples/second is achieved, but for the GTX 1080Ti, a speed of 10 Gsamples/second is achieved. 

### GeForce GTX 1050 

	Mask generated in 24.94 seconds
	Formatting...
	Total iterations: 2.18473e+10
	Iterations per pixel: 5926.45
	Maximum frequency: 68115
	Minimum frequency: 0
	Total time: 29.19s
	Iterations per second: 7.48e+08

### GeForce GTX 1080Ti

	Mask generated in 14.47 seconds
	Formatting...
	Total iterations: 3.49555e+11
	Iterations per pixel: 94822.95
	Maximum frequency: 1086572
	Minimum frequency: 0
	Total time: 32.98s
	Iterations per second: 1.06e+10