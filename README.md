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

<p align="center">
	<img src="/images/example_grainy_20.png" width="30%" />
	<img src="/images/example_fine_20.png" width="30%" />
</p>

### 100 iteration count

<p align="center">
	<img src="/images/example_grainy_100.png" width="200" />
	<img src="/images/example_fine_100.png" width="200" />
</p>
