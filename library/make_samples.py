#!/bin/env python3
import os
import time
import numpy as np
import numpy.random as rr

WDIR = '../datasets'

# Sample size
size = 15000

# Create three seeds
s1 = int(time.time())
s2 = int(np.log(time.time()) * 1e4)
s3 = int(np.log10(time.time()) * 1e4)
s4 = int(np.log2(time.time()) * 1e4)

# Produce three uniform samples
for i, seed in enumerate([s1, s2, s3]):
    rr.seed(seed)
    sample = rr.rand(size, 2) * 2 - 1

    f = open(os.path.join(WDIR, '01_Aleatoreidad_muestra{}.txt'.format(i+1)),
    'w')
    f.write('#seed {}'.format(seed))

    np.savetxt(f, sample, delimiter=',')

# Produce false sample
nsample = np.empty([0, 2])

rr.seed(s4)
while(len(nsample)) < size:
    nsamplei = rr.randn(size , 2) * 2.0 + [0.5, 0.5]

    # Keep only samples in original region
    cond = (np.abs(nsamplei[:, 0]) <=1) * (np.abs(nsamplei[:, 1]) <=1)
    nsamplei = np.compress(cond, nsamplei, axis=0)

    nsample = np.vstack([nsample, nsamplei])

# Write sample
f = open(os.path.join(WDIR, '01_Aleatoreidad_muestra{}.txt'.format(i+2)),
'w')
f.write('#seed {}'.format(seed))
np.savetxt(f, nsample, delimiter=',')