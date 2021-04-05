import sys

sys.path.append('/mnt/zfsusers/rstiskalek/pysham/')

from PySHAM import mocks

# Arguments defining the AM model
args = {'nd_gal': '/mnt/zfsusers/rstiskalek/pysham/results/NYUmatch/LF.npy',
        'scope': (-21.0, -20.0),
        'halos': '/mnt/zfsusers/rstiskalek/pysham/data/halos_mvir_proxy.npy',
        'boxsize': 400,
        'halos_proxy': mocks.VirialMassProxy(),
        'max_scatter': 1.0,
        'Nmocks': 50,
        'nthreads': 1}

theta = {'alpha': 1.1, 'scatter': 0.2}

AM_model = mocks.AbundanceMatch(**args)

print('We have the AM model')


#matched_samples = AM_model.match(theta)

# Save...



