# import torch
# from torchvision.utils import save_image
import time
import sys
import os
import scipy.misc
# import numpy as np
import cupy as np

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_random_points(x_min, x_max, y_min, y_max, n_points):
	# s_p = torch.rand(n_points, 2)
	s_p = np.random.random((n_points, 2))
	s_p[:,0] *= y_max - y_min + 2
	s_p[:,1] *= x_max - x_min + 2
	s_p[:,0] += y_min - 1
	s_p[:,1] += x_min - 1
	# return s_p, s_p.clone()
	return s_p, s_p.copy()

def filter_by_distance(s_p, i_p):
	dists = get_absolute_values(i_p)
	return s_p[dists > 9, :]

def get_absolute_values(points):
	return np.sum(abs(points), axis=-1)
	points[points > 10] = 10
	points[points != points] = 10
	points = points * points
	dists = points[:,0] + points[:,1]
	# points[:,0] += points[:,1]

	# dists = points[:,0]
	# torch.norm(points, dim=1, out = dists)
	# dists = dists*dists
	return points[:,0]

def first_check(x_min, x_max, y_min, y_max, iters, n_points):
	s_p, i_p = get_random_points(x_min, x_max, y_min, y_max, n_points)
	for i in range(iters):
		# dists = (i_p*i_p).sum(1)
		# dists[dists != dists] = 100
		i_p[:,0], i_p[:,1] = (
			i_p[:,0]*i_p[:,0] - i_p[:,1]*i_p[:,1] + s_p[:,0],
			2*i_p[:,0]*i_p[:,1] + s_p[:,1]
		)
		# i_p[dists <= 9,0], i_p[dists <= 9,1] = (
			# i_p[dists <= 9,0]*i_p[dists <= 9,0] - i_p[dists <= 9,1]*i_p[dists <= 9,1] + s_p[dists <= 9,0],
			# 2*i_p[dists <= 9,0]*i_p[dists <= 9,1] + s_p[dists <= 9,1]
		# )
	s_p = filter_by_distance(s_p, i_p)
	# return s_p, s_p.clone()
	return s_p, s_p.copy()

def from_pixels(pixels, x_min, x_max, x_dim, y_min, y_max, y_dim):
	pass

def to_pixels(points, x_min, x_max, x_dim, y_min, y_max, y_dim):
		# pixels = points.clone()
		pixels = points.copy()
		pixels[:,0] -= y_min
		pixels[:,1] -= x_min
		pixels[:,0] /= y_max - y_min
		pixels[:,1] /= x_max - x_min
		pixels[:,0] *= y_dim
		pixels[:,1] *= x_dim
		# pixels = pixels.long()
		pixels = pixels.astype(np.int32)
		pixels = pixels[
			(pixels[:,0] >= 0)
			& (pixels[:,1] >= 0)
			& (pixels[:,0] < y_dim)
			& (pixels[:,1] < x_dim)
		]
		return pixels

def iterate(buddha_tensor, x_min, x_max, x_dim,
	y_min, y_max, y_dim, iters, sample_size):
	n_points = 2**sample_size
	s2, i2 = first_check(x_min, x_max, y_min, y_max, iters, n_points)

	if len(s2) == 0:
		return

	for i in range(iters):
		i2[:,0], i2[:,1] = (
			i2[:,0]*i2[:,0] - i2[:,1]*i2[:,1] + s2[:,0],
			2*i2[:,0]*i2[:,1] + s2[:,1]
		)

		dists = get_absolute_values(i2)
		s2 = s2[dists < 25, :]
		i2 = i2[dists < 25, :]
		if len(i2) == 0:
			break

		pixels = to_pixels(i2, x_min, x_max, x_dim, y_min, y_max, y_dim)
		if len(pixels) == 0:
			break
		buddha_tensor[pixels[:,0], pixels[:,1]] += 1

def make_mandelbrot(x_min, x_max, x_dim, y_min, y_max, y_dim,
	iters, gens, sample_size):
	t0 = time.time()
	# buddha_tensor = torch.zeros(y_dim, x_dim).float()
	buddha_tensor = np.zeros((y_dim, x_dim), dtype=np.float32)
	for i in range(gens):
		iterate(buddha_tensor, x_min, x_max, x_dim,
			y_min, y_max, y_dim, iters, sample_size)
		print("\r\titeration: %d/%d" % (i+1, gens), end='', flush=True)


	t1 = time.time()
	print_stats(buddha_tensor, x_dim, y_dim, t1-t0)
	s = buddha_tensor.sum()
	buddha_tensor /= buddha_tensor.max()
	# buddha_tensor = np.min(1.1*buddha_tensor, buddha_tensor*.2+.8)

	# save_image(
		# buddha_tensor,
		# "buddha_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, s)
	# )
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

	iters, gens, sample_size = read_sys_args()

	make_mandelbrot(x_min, x_max, x_dim, y_min, y_max, y_dim,
		iters, gens, sample_size)
	print()

