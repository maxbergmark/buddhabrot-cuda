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

![Grainy image 20](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/example_grainy_20.png) ![Fine image 20](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/example_fine_20.png)

### 100 iteration count

![Grainy image 100](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/example_grainy_100.png) ![Fine image 100](https://raw.githubusercontent.com/maxbergmark/buddhabrot-cuda/master/images/example_fine_100.png)