# Copyright (C) 2020  Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""MPI submission script."""
import os

import numpy as np

import argparse
from mpi4py import MPI

from posterior import Posterior
from PySHAM import utils

# setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()          # Labels the individual processes/cores
size = comm.Get_size()          # Total number of processes/cores

# read the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--config", help="Config file path", type=str)
parser.add_argument("--nd_type", help="LF or SMF", type=str)
parser.add_argument("--bin_index", help="Index of the bin", type=int)
parser.add_argument("--file_index", help="Index of the run.", type=int)
parser.add_argument("--cached", default=False, action='store_true',
                    help="Load cached points.")
args = parser.parse_args()

posterior = Posterior(args)

PROXY = posterior.model_likelihood.AM_model.halo_proxy.name

TEMP_FNAME = './results/{}/{}_AM_nd{}_proxy{}_bin{}_file{}.p'
SUB_FNAME = './data/submit_points/AM_name{}_nd{}_proxy{}_bin{}_file{}.p'
OUT_FNAME = './results/{}/AM_nd{}_proxy{}_bin{}_file{}.p'

# print basic info to the out file
if rank == 0:
    print('Running {} with {} bin index {}.'.format(args.name, args.nd_type,
                                                    args.bin_index))
    if args.cached:
        print('Running with cached points')
    else:
        print('Estimated a grid')

if args.cached:
    # load cached points
    fname = SUB_FNAME.format(args.name, args.nd_type, PROXY, args.bin_index,
                             args.file_index)
    X = utils.load_pickle(fname)
else:
    # grid search inside the boundaries
    pars = posterior.model_likelihood.parameters
    nside = posterior.nside
    bounds = posterior.bounds
    points = np.meshgrid(*[np.linspace(bounds[p][0], bounds[p][1], nside[p])
                         for p in pars])
    X = np.vstack([p.reshape(-1,) for p in points]).T
    print('X shape is {}'.format(X.shape))


def sample_point(index):
    """Samples a point at position ``X[index, :]`` and saves the results."""
    pars = posterior.model_likelihood.parameters
    theta = {p: X[index, i] for i, p in enumerate(pars)}
#    print(theta)
    fname = TEMP_FNAME.format(args.name, index, args.nd_type, PROXY,
                              args.bin_index, args.file_index)
    blobs = posterior(theta)
    utils.dump_pickle(fname, blobs)
    print('Finished {}'.format(index))


Njobs = X.shape[0]
N_per_proc = int(np.floor(Njobs / size))

# Do the sampling
if rank != size - 1:
    for i in range(N_per_proc):
        sample_point(rank * N_per_proc + i)
else:
    for i in range(N_per_proc * rank, Njobs):
        sample_point(i)

# This causes threads to wait until they've all finished
buff = np.zeros(1)
if rank == 0:
    for i in range(1, size):
        comm.Recv(buff, source=i)
else:
    comm.Send(buff, dest=0)

# Collect outputs
if rank == 0:
    out = [None] * Njobs
    for i in range(Njobs):
        fname = TEMP_FNAME.format(args.name, i, args.nd_type, PROXY,
                                  args.bin_index, args.file_index)
        out[i] = utils.load_pickle(fname)
        os.remove(fname)

    fname = OUT_FNAME.format(args.name, args.nd_type, PROXY, args.bin_index,
                             args.file_index)
    utils.dump_pickle(fname, out)
    # try to remove the submit file
    try:
        fname = SUB_FNAME.format(args.name, args.nd_type, PROXY,
                                 args.bin_index, args.file_index)
        os.remove(fname)
    except FileNotFoundError:
        pass

    print('All finished')
