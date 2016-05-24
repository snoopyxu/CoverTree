import sys
import numpy as np
import torchfile

def convert_to_cov_format(in_arr, filepath, count):
	with open(filepath, 'w') as f:
		if in_arr.dtype != np.dtype('float64'):
			in_arr = in_arr.astype('float64')
			if count != 'all':
				in_arr = in_arr[:int(count)]
		n_points = np.array(in_arr.shape[0], dtype='int32')
		n_points.tofile(f)
		dims = np.array(in_arr.shape[1], dtype='int32')
		dims.tofile(f)
		in_arr.tofile(f)

if 3 < len(sys.argv) < 4:
	sys.argv.append('all')
elif len(sys.argv) < 3:
    print "Use: python convert.py <original file> <output filename> <all|number of features to convert>"
in_arr = torchfile.load(sys.argv[1])
convert_to_cov_format(in_arr, sys.argv[2], sys.argv[3])	
