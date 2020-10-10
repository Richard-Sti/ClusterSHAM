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
from sys import path

import numpy as np

import argparse
from mpi4py import MPI


from posterior import Posterior
path.append('../')
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
parser.add_argument("--npoints", type=int, help="Number of points to sample.",
                    default=None)
parser.add_argument("--cached", type=bool, help="Load cached points.")
args = parser.parse_args()

posterior = Posterior(args)

#NAME = posterior.info['name']
#PROXY = posterior.AM_model.halo_proxy.name
#SCOPE = posterior.AM_model.scope
#FILE_INDEX = posterior.info['file_index']
#
## print basic info to the out file
#if rank == 0:
#    print('Running {} with {} from {} to {}'.format(NAME, PROXY, *SCOPE))
#
#if rank == 0 and args.cached:
#    print('Running with cached points')
#else:
#    print('Estimated a grid')
#
#if args.cached:
#    # load cached points
#    X = utils.load_pickle('./data/submit_points/AM_{}_{}_{}_{}_{}.p'.format(
#            NAME, FILE_INDEX, PROXY, *SCOPE))
#else:
#    # grid search over the boundaries
#    pars = posterior.model_likelihood.parameters
#    bounds = posterior.bounds
#    nside = int(args.npoints**(1 / len(pars)))
#    points = np.meshgrid(*[np.linspace(bounds[p][0], bounds[p][1], nside)
#                         for p in pars])
#    X = np.vstack([p.reshape(-1,) for p in points]).T
#
#
#def sample_point(index):
#    """Samples a point at position ``X[index, :]`` and saves the results."""
#    pars = posterior.model_likelihood.parameters
#    theta = {p: X[index, i] for i, p in enumerate(pars)}
#    blobs = posterior(theta)
#
#    fname = './results/{}/{}_AM_{}_{}_{}_{}.p'.format(NAME, index, PROXY,
#                                                      FILE_INDEX, *SCOPE)
#    utils.dump_pickle(fname, blobs)
#    print('Finished {}'.format(index))
#
#
#Njobs = X.shape[0]
#N_per_proc = int(np.floor(Njobs / size))
#
## Do the sampling
#if rank != size - 1:
#    for i in range(N_per_proc):
#        sample_point(rank * N_per_proc + i)
#else:
#    for i in range(N_per_proc * rank, Njobs):
#        sample_point(i)
#
## This causes threads to wait until they've all finished
#buff = np.zeros(1)
#if rank == 0:
#    for i in range(1, size):
#        comm.Recv(buff, source=i)
#else:
#    comm.Send(buff, dest=0)
#
## Collect outputs
#if rank == 0:
#    out = [None] * Njobs
#    for i in range(Njobs):
#        fname = './results/{}/{}_AM_{}_{}_{}_{}.p'.format(NAME, i, PROXY,
#                                                          FILE_INDEX, *SCOPE)
#        out[i] = utils.load_pickle(fname)
#        os.remove(fname)
#
#    out_fname = './results/{}/AM_{}_{}_{}_{}.p'.format(NAME, PROXY, FILE_INDEX,
#                                                       *SCOPE)
#    utils.dump_pickle(out_fname, out)
#    # try to remove the submit file
#    try:
#        os.remove('./data/submit_points/AM_{}_{}_{}_{}_{}.p'.format(
#            NAME, FILE_INDEX, PROXY, *SCOPE))
#    except FileNotFoundError:
#        pass
#
#    print('All finished')
