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

![Grainy image](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/pycuda_1440x2560_20_128.png)

![Grainy image](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/pycuda_1440x2560_20_256.png)