import numpy as np
import h5py
import sys

filename = sys.argv[1]
num_chunks = int(sys.argv[2])

data = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data.append(line.strip().encode('utf-8'))

#Creating HDF5 dataset
hf = h5py.File(filename + '.h5', 'w')
hf.create_dataset('dataset', data=np.array(data), chunks=(len(data) // num_chunks, ))

hf.close()
