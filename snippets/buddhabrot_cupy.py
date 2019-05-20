import time
import sys
import os
import scipy.misc
# import numpy as np
import cupy as np
# from numba import vectorize, float64
# import numba

abs2 = np.ElementwiseKernel(
	'complex64 x',
	'float32 y',
	'y = isfinite(x) ? x.real()*x.real() + x.imag()*x.imag() : 100',
	'abs2',
	False
)
abs2.__call__(np.array([1+1j]).astype(np.complex64))
np.nonzero(np.array([1]).astype(np.float32))
map_dists_greater = np.ElementwiseKernel(
	'float32 x',
	'bool y',
	'y = x > 4',
	'map_dists_greater',
	False
)
map_dists_greater.__call__(np.array([1]).astype(np.float32))

map_dists_less = np.ElementwiseKernel(
	'float32 x',
	'bool y',
	'y = x < 25',
	'map_dists_less',
	False
)
map_dists_less.__call__(np.array([1]).astype(np.float32))

def get_random_points(x_min, x_max, y_min, y_max, n_points):
	s_p = np.random.random((n_points, 2), dtype=np.float32)
	s_p[:,0] *= y_max - y_min + 0
	s_p[:,1] *= x_max - x_min + 2
	s_p[:,0] += y_min - 0
	s_p[:,1] += x_min - 1
	# s_p[:,0] *= 3.0
	# s_p[:,1] *= 3.5
	# s_p[:,0] -= 1.5
	# s_p[:,1] -= 2.5
	complex_numbers = (s_p[:,0]*1j + s_p[:,1]).astype(np.complex64)
	return complex_numbers, complex_numbers



def get_absolute_values(points):
	points = abs2(points)
	return points


def filter_by_distance_greater(s_p, i_p):
	# print(i_p.device)
	# mask2 = np.ones_like(mask).astype(np.bool)
	# print(mask, mask.dtype)
	# print(mask2, mask2.dtype)
	# print(np.equal(mask, mask2))
	# t0 = time.time()
	dists = get_absolute_values(i_p)
	# t1 = time.time()
	# s_p = s_p[dists > 4]
	# dists = dists > 4
	dists = map_dists_greater(dists)
	indices = np.nonzero(dists)
	# mask = np.nonzero(dists > 4)
	# t2 = time.time()
	s_p = s_p[indices]
	# t3 = time.time()
	# print()
	# print("get_absolute_values: %.4f\ncreate mask: %.4f\nfilter array: %.4f\nnumber of elements: %d" % (t1-t0, t2-t1, t3-t2, np.sum(dists > 4)))
	# return s_p
	return s_p

def filter_by_distance_less(s_p, i_p):
	t0 = time.time()
	dists = get_absolute_values(i_p)
	t1 = time.time()
	mask = map_dists_less(dists)
	indices = np.nonzero(mask)
	s_p = s_p[indices]
	i_p = i_p[indices]
	t2 = time.time()
	# print()
	# print("%.4f\t%.4f\t%d" % (t1-t0, t2-t1, np.sum(dists < 25)))
	return s_p, i_p


def first_check(x_min, x_max, y_min, y_max, iters, n_points):
	s_p, i_p = get_random_points(x_min, x_max, y_min, y_max, n_points)
	for i in range(iters):
		i_p = i_p*i_p + s_p
	s_p = filter_by_distance_greater(s_p, i_p)

	return s_p, s_p

def to_pixels(points, x_min, x_max, x_dim, y_min, y_max, y_dim):
		pixels = np.vstack((points.real, points.imag)).T
		pixels[:,0] -= y_min
		pixels[:,1] -= x_min
		pixels[:,0] /= y_max - y_min
		pixels[:,1] /= x_max - x_min
		pixels[:,0] *= y_dim
		pixels[:,1] *= x_dim
		pixels = pixels.astype(np.int32)
		pixels = pixels[
			(pixels[:,0] >= 0)
			& (pixels[:,0] < y_dim)
			& (pixels[:,1] >= 0)
			& (pixels[:,1] < x_dim)
		]
		return pixels

def iterate(buddha_tensor, x_min, x_max, x_dim,
	y_min, y_max, y_dim, iters, sample_size):
	n_points = 2**sample_size
	s2, i2 = first_check(x_min, x_max, y_min, y_max, iters, n_points)

	if len(s2) == 0:
		return

	mask = s2.real < 100
	for i in range(iters):
		i2 = i2*i2 + s2
		s2, i2 = filter_by_distance_less(s2, i2)
		# next_value(i2, s2, mask)

		if len(i2) == 0:
			break

		pixels = to_pixels(i2, x_min, x_max, x_dim, y_min, y_max, y_dim)
		if len(pixels) == 0:
			break
		buddha_tensor[pixels[:,0], pixels[:,1]] += 1


def make_buddhabrot(x_min, x_max, x_dim, y_min, y_max, y_dim,
	iters, gens, sample_size):
	t0 = time.time()
	buddha_tensor = np.zeros((y_dim, x_dim), dtype=np.float32)
	for i in range(gens):
		iterate(buddha_tensor, x_min, x_max, x_dim,
			y_min, y_max, y_dim, iters, sample_size)
		print("\r\titeration: %d/%d" % (i+1, gens), end='', flush=True)

	t1 = time.time()
	print_stats(buddha_tensor, x_dim, y_dim, t1-t0)

	s = buddha_tensor.sum()
	buddha_tensor /= buddha_tensor.max()
	buddha_tensor = np.minimum(1.1*buddha_tensor, buddha_tensor*.2+.8)

	scipy.misc.toimage(np.asnumpy(buddha_tensor), cmin=0.0, cmax=1.0).save(
		"buddha_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, s)
	)

def print_stats(buddha_tensor, x_dim, y_dim, elapsed_time):
	s = buddha_tensor.sum()
	max_freq = buddha_tensor.max()
	min_freq = buddha_tensor.min()
	mean_freq = s / (x_dim*y_dim)
	print("\n")
	print("\tImage of size %dx%d generated" % (x_dim, y_dim))
	print("\tTotal iterations: %.2e" % s)
	print("\tIterations per pixel: %.2f" % mean_freq)
	print("\tMinimum frequency: %d" % min_freq)
	print("\tMaximum frequency: %d" % max_freq)
	print("\tElapsed time: %.2fs" % elapsed_time)
	print("\tIterations per second: %.2e" % (s / elapsed_time,))

def print_usage():
	print("\tUsage: python %s [options]" % (os.path.basename(__file__),))
	print()
	print("\t  -i [0-9]+\tnumber of iterations per sample")
	print("\t  -g [0-9]+\tnumber of generations")
	print("\t  -n [0-9] \tlog_2 of number of samples per generation")

def read_sys_args():
	if '-h' in sys.argv:
		print_usage()
		quit()
	if '-i' in sys.argv:
		idx = sys.argv.index('-i')
		try:
			iters = int(sys.argv[idx+1])
		except:
			print_usage()
			quit()
	else:
		print("\tRunning with default 20 iterations")
		iters = 20
	if '-g' in sys.argv:
		idx = sys.argv.index('-g')
		try:
			gens = int(sys.argv[idx+1])
		except:
			print_usage()
			quit()
	else:
		print("\tRunning with default 10 generations")
		gens = 10
	if '-n' in sys.argv:
		idx = sys.argv.index('-n')
		try:
			sample_size = int(sys.argv[idx+1])
		except:
			print_usage()
			quit()
	else:
		print("\tRunning with default 2^20 batch size")
		sample_size = 20
	return iters, gens, sample_size

if __name__ == "__main__":
	print()
	y_min, y_max = -3.2, 2.0
	x_min, x_max = -1.5, 1.5
	x_dim, y_dim = 1440, 2560
	# y_min, y_max = -2.2, 1.0
	# x_min, x_max = -1.5, 1.5
	# x_dim, y_dim = 1000, 1000

	iters, gens, sample_size = read_sys_args()

	make_buddhabrot(x_min, x_max, x_dim, y_min, y_max, y_dim,
		iters, gens, sample_size)
	print()

