import os

import numpy as np

from hp.rotator import angdist
from pymangle import Mangle
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Labels the individual processes/cores
size = comm.Get_size()  # Total number of processes/cores

m = Mangle("/mnt/zfsusers/rstiskalek/pysham/data/tmp_lss_geometry.ply")
RA, DEC = m.genrand_range(320000, 100, 260, -10, 66)

# Remove objects near some odd areas of the survey geometry
theta = np.pi/2 - np.deg2rad(DEC)
phi = np.deg2rad(RA)
dist1 = angdist(np.vstack([theta, phi]), [0.16*np.pi, 1.44*np.pi])
dist2 = angdist(np.vstack([theta, phi]), [0.51*np.pi, 1.38*np.pi])
IDS = np.intersect1d(np.where(dist1 > 0.15), np.where(dist2 > 0.08))
output = np.vstack([RA[IDS], DEC[IDS]]).T

out_file = 'out_' + str(rank) + '.dat'
np.savetxt(out_file, output)
# This causes threads to wait until they've all finished
buff = np.zeros(1)
if rank == 0:
    for i in range(1, size):
        comm.Recv(buff, source=i)
else:
    comm.Send(buff, dest=0)

# At end, a single thread does things like concatenate the files
if rank == 0:
    string = 'cat `find ./ -name "out_*" | sort -V` > out.dat'
    os.system(string)
    string = 'rm out_*.dat'
    os.system(string)

    out = np.loadtxt("out.dat")
    os.system("rm out.dat")

    RA = out[:, 0]
    DEC = out[:, 1]
    Nmock = RA.size
    random_catalog = np.zeros(Nmock, dtype={'names': ('RA', 'DEC'),
                                            'formats': ['float64'] * 2})
    random_catalog['RA'] = np.ravel(RA)
    random_catalog['DEC'] = np.ravel(DEC)

    np.save('../Data/RandCatNYU.npy', random_catalog)
