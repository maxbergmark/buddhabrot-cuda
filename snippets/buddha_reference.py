#! /usr/bin/python
import png
import numpy as np
from random import random

def c_set(num_samples, iterations):
	# return a sampling of complex points outside of the mset

	# Allocate an array to store our non-mset points as we find them.
	non_msets = np.zeros(num_samples, dtype=np.complex128)
	non_msets_found = 0

	# create an array of random complex numbers (our 'c' points)
	c = (np.random.random(num_samples)*4-2 + \
		(np.random.random(num_samples)*4-2)*1j)

	# Optimizations: most of the mset points lie within the
	# within the cardioid or in the period-2 bulb. (The two most
	# prominant shapes in the mandelbrot set. We can eliminate these
	# from our search straight away and save alot of time.
	# see: http://en.wikipedia.org/wiki/Mandelbrot_set#Optimizations

	print "%d random c points chosen" % len(c)
	# First elimnate points within the cardioid
	p = (((c.real-0.25)**2) + (c.imag**2))**.5
	c = c[c.real > p- (2*p**2) + 0.25]
	print "%d left after filtering the cardioid" % len(c)

	# Next eliminate points within the period-2 bulb
	c = c[((c.real+1)**2) + (c.imag**2) > 0.0625]
	print "%d left after filtering the period-2 bulb" % len(c)

	# optimizations done.. time to do the escape time algorithm.
	# Use these c-points as the initial 'z' points.
	# (saves one iteration over starting from origin)
	z = np.copy(c)

	for i in range(iterations):
		# apply mandelbrot dynamic
		z = z ** 2 + c

		# collect the c points that have escaped
		mask = abs(z) < 2
		new_non_msets = c[mask == False]
		non_msets[non_msets_found:non_msets_found+len(new_non_msets)]\
				  = new_non_msets
		non_msets_found += len(new_non_msets)

		# then shed those points from our test set before continuing.
		c = c[mask]
		z = z[mask]

		# print "iteration %d: %d points have escaped!"\
		# % (i + 1, len(new_non_msets))

	# return only the points that are not in the mset
	return non_msets[:non_msets_found]

def buddhabrot(c, size):
	# initialise an empty array to store the results
	img_array = np.zeros([size, size], int)

	# use these c-points as the initial 'z' points.
	z = np.copy(c)

	while(len(z)):

		# print "%d orbits in play" % len(z)
		# translate z points into image coordinates
		x = np.array((z.real + 2.) / 4 * size, int)
		y = np.array((z.imag + 2.) / 4 * size, int)

		# add value to all occupied pixels
		img_array[x, y] += 1

		# apply mandelbrot dynamic
		z = z ** 2 + c

		# shed the points that have escaped
		mask = abs(z) < 2
		c = c[mask]
		z = z[mask]

	return img_array

if __name__ == "__main__":

	size = 1000 # size of final image
	iterations = 20 # bailout value -- higher means more details
	samples = 10000000 # number of random c points chosen

	img_array = np.zeros([size, size], int)

	i = 0

	while True:

		print "get c set..."
		c = c_set(samples, iterations)
		print "%d non-mset c points found." % len(c)

	print "render buddha..."
	img_array += buddhabrot(c, size)

	print "adjust levels..."
	e_img_array = np.array(img_array/float(img_array.max())*((2**16)-1), int)

	print "saving buddhabrot_n_%di_%03d.png" % (iterations,i)
	# save to final render to png file
	imgWriter = png.Writer(size, size, greyscale=True, alpha=False, bitdepth=16)
	f = open("buddhabrot_n_%di_%03d.png" % (iterations,i), "wb")
	imgWriter.write(f, e_img_array)
	f.close()

	print "Done."
	i += 1