import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Device
from pycuda import gpuarray
import time
import scipy.misc
from matplotlib import pyplot as plt
import os

code = open("buddha_kernel.cu", "r").read()

def print_stats(cpu_canvas, elapsed_time, x_dim, y_dim):
	total_iterations = np.sum(cpu_canvas)
	max_freq = np.max(cpu_canvas)
	min_freq = np.min(cpu_canvas)
	print("\tTotal iterations: %.5e" % total_iterations)
	print("\tIterations per pixel: %.2f" % (total_iterations / (x_dim*y_dim),))
	print("\tMaximum frequency: %d" % max_freq)
	print("\tMinimum frequency: %d" % min_freq)
	print("\tTotal time: %.2fs" % (elapsed_time,))
	print("\tIterations per second: %.2e" % (total_iterations / (elapsed_time),))

def format_and_save(cpu_canvas, x_dim, y_dim, threads, iters):
	cpu_canvas /= max(1, np.max(cpu_canvas))
	cpu_canvas.shape = (y_dim, x_dim)
	# this just makes the color gradient more visually pleasing
	cpu_canvas = np.minimum(2.5*cpu_canvas, cpu_canvas*.2+.8)

	file_name = "images/pycuda_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	print("\n\tSaving %s..." % file_name)
	scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(file_name)
	print("\tImage saved!\n")

def transform_image(canvas, x_dim, y_dim, dim):
	print("\tFormatting...")
	if dim == 1:
		canvas.shape = (y_dim, x_dim)
		return canvas
	new = np.zeros((y_dim, x_dim), dtype = np.float64)
	for i in range(0, y_dim*x_dim, dim*dim):
		block = canvas[i:i+dim*dim]
		block.shape = (dim,dim)
		block_row = (i // (dim*x_dim)) * dim
		block_col = (i // dim) % x_dim
		new[block_row:block_row+dim, block_col:block_col+dim] = block
		# new[block_col:block_col+dim, block_row:block_row+dim] = block
	return new

def generate_image(x_dim, y_dim, iters):

	threads = 2**7
	b_s = 2**10
	dim = 16
	disc = np.int32(1024)
	grid_size = np.float32(1 / disc)
	repeat = 1

	device = Device(0)
	print("\n\t" + device.name(), "\n")
	context = device.make_context()

	formatted_code = code % {
		"XDIM" : x_dim,
		"YDIM" : y_dim,
		"ITERS" : iters,
		"DIM" : dim,
		"REPEAT" : repeat
	}

	# generate kernel and setup random number generation
	module = SourceModule(
		formatted_code,
		no_extern_c = True,
		options = ['--use_fast_math', '-O3', '--ptxas-options=-O3', '-I%s/include' % os.getcwd()]
	)
	mask_func = module.get_function("mask_kernel")
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	max_mask = gpuarray.zeros(disc * disc, dtype = np.uint32)
	min_mask = gpuarray.zeros(disc * disc, dtype = np.uint32) + iters
	canvas = gpuarray.zeros(y_dim * x_dim, dtype = np.uint32)

	t0 = time.time()
	mask_func(min_mask, max_mask, seed, grid_size, disc, 
		block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t1 = time.time()
	print("\tMask generated in %.2f seconds" % (t1-t0,))
	# cpu_mask = min_mask.get()
	# print(min_mask, min_mask.get().max())
	# print(max_mask)

	t0 = time.time()
	use_mask = np.uint32(1)
	fill_func(canvas, seed, grid_size, disc, min_mask, max_mask, use_mask, 
		block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t1 = time.time()

	# fetch buffer from gpu and save as image
	cpu_canvas = canvas.get().astype(np.float64)
	cpu_canvas = transform_image(cpu_canvas, x_dim, y_dim, dim)
	context.pop()
	elapsed_time = t1-t0

	# cpu_mask.shape = (disc, disc)
	# print(cpu_mask.min())
	# plt.imshow(cpu_mask, interpolation='nearest')
	# plt.show()

	print_stats(cpu_canvas, elapsed_time, x_dim, y_dim)
	format_and_save(cpu_canvas, x_dim, y_dim, threads, iters)

def run_benchmark(x_dim, y_dim, iters, dim, threads, b_s, disc, repeat):
	grid_size = np.float32(1 / disc)

	device = Device(0)
	print("\n\t" + device.name(), "\n")
	context = device.make_context()

	formatted_code = code % {
		"XDIM" : x_dim,
		"YDIM" : y_dim,
		"ITERS" : iters,
		"DIM" : dim,
		"REPEAT" : repeat
	}

	# generate kernel and setup random number generation
	module = SourceModule(
		formatted_code,
		no_extern_c=True,
		options=['--use_fast_math', '-O3', '--ptxas-options=-O3', '-I./']
	)
	mask_func = module.get_function("mask_kernel")
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	max_mask = gpuarray.zeros(disc * disc, dtype = np.uint32)
	min_mask = gpuarray.zeros(disc * disc, dtype = np.uint32) + iters
	canvas = gpuarray.zeros(y_dim * x_dim, dtype = np.uint32)

	t0 = time.time()
	mask_func(min_mask, max_mask, seed, grid_size, disc, 
		block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t1 = time.time()
	print("\tMask generated in %.2f seconds" % (t1-t0,))

	mask_time = 0
	non_mask_time = 0
	for i in range(40):
		t0 = time.time()
		use_mask = np.uint32(1)
		fill_func(canvas, seed, grid_size, disc, min_mask, max_mask, use_mask, 
			block=(b_s,1,1), grid=(threads,1,1))
		context.synchronize()
		t1 = time.time()
		mask_time += t1-t0

		t0 = time.time()
		use_mask = np.uint32(0)
		fill_func(canvas, seed, grid_size, disc, min_mask, max_mask, use_mask, 
			block=(b_s,1,1), grid=(threads,1,1))
		context.synchronize()
		t1 = time.time()
		non_mask_time += t1-t0

	context.pop()
	print("\tWith mask: %.2fs" % mask_time)
	print("\tWithout mask: %.2fs" % non_mask_time)
	print("\tSpeedup: %.2f" % (non_mask_time / mask_time,))
	return mask_time, non_mask_time

def run_test_suite(x_dim, y_dim, iters):
	mask_times = []
	non_mask_times = []

	dim = 16
	threads = 2**7
	b_s = 2**7
	disc = 16
	repeat = 1

	measurement = [1, 2, 4, 8, 16, 32, 64, 128]
	# measurement_long = [2**4, 2**5, 2**6, 2**7, 2**8]

	# for dim in measurement:
	for disc in measurement:
	# for repeat in measurement:
	# for b_s in measurement:
	# for threads in measurement:
		mask_time, non_mask_time = run_benchmark(x_dim, y_dim, iters, dim, threads, b_s, np.int32(disc), repeat)
		# mask_times.append(mask_time / disc**2)
		# non_mask_times.append(non_mask_time / disc**2)
		mask_times.append(mask_time / disc**2 / repeat / b_s / threads)
		non_mask_times.append(non_mask_time / disc**2 / repeat / b_s / threads)
	# print(mask_times)
	# print(non_mask_times)
	plt.loglog(measurement, mask_times, '*', label = "Mask times")
	plt.loglog(measurement, non_mask_times, '*', label = "Non mask times")
	plt.legend()
	plt.show()

if __name__ == "__main__":

	x_dim = 1440
	y_dim = 2560
	iters = 20
	generate_image(x_dim, y_dim, iters)
	# run_test_suite(x_dim, y_dim, iters)