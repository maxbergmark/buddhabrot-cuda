import numpy as np
import scipy.misc

my_data = np.genfromtxt('example.txt', delimiter=',').astype(np.float32)
my_data /= np.max(my_data)
scipy.misc.toimage(my_data, cmin=0.0, cmax=1.0).save("cpp.png")
