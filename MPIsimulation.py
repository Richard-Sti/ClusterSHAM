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

from PySHAM import (mocks, utils)


# Create shared array halos
comm = MPI.COMM_WORLD
rank = comm.Get_rank()          # Labels the individual processes/cores
size = comm.Get_size()          # Total number of processes/cores

# ---------------------------------------- #
#       Parse the input arguments          #
# ---------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to the config file.")
parser.add_argument("--name", help="Catalog name", type=str)


args = parser.parse_args()
config_path = toml.load(args.config)

# parse the arguments
args_AM = utils.abundance_matching_parser(config_path)
args_likelihood = utils.clustering_likelihood_parser(config_path)

AM_model = mocks.AbundanceMatch(**args_AM)
args_likelihood.update({'AM_model': AM_model})
# create the model
model = mocks.ClusteringLikelihood(**args_likelihood)

if rank == 0:
    print('Running {} with {} from {} to {}'
          .format(args.name, AM_model.halo_proxy.name, *AM_model.scope))






#  -------------------------------------------------------------------------- #
#                   Get variables from config                                 #
# --------------------------------------------------------------------------- #

rpmin, rpmax, nrpbins, boxsize, subside, pimax, nthreads,\
        Nmocks, npoints = [config['main'][p]
                           for p in ['rpmin', 'rpmax', 'nrpbins', 'boxsize',
                                     'subside', 'pimax', 'nthreads', 'Nmocks',
                                     'npoints']]
pars = config[args.name]['pars']
if not args.cached:
    bounds = {'alpha': (args.alpha[0], args.alpha[1]),
              'scatter': (args.scatter[0], args.scatter[1])}
else:
    bounds = {'alpha': (-5., 5.),
              'scatter': (0.005, 1.)}
if args.scope[0] > 0:
    AMtype = 'MF'
else:
    AMtype = 'LF'
nd_gal = np.load(config[args.name][AMtype]['nd_gal'])
rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
if args.proxy == 'vvir':
    proxy = utils.vvir_proxy
elif args.proxy == 'mvir':
    proxy = utils.mvir_proxy
halos = np.load('./data/halos_{}.npy'.format(args.proxy))
survey_wp = utils.load_pickle('./results/{}/ObsCF{}to{}.p'.format(
    args.name, args.scope[0], args.scope[1]))
# get the model
model = Model(args.name, pars, args.scope, proxy, nd_gal, halos, bounds,
              survey_wp, boxsize, subside, rpbins, pimax, nthreads, Nmocks)

if not args.cached:
    nside = int(npoints**(1 / len(pars)))
    points = np.meshgrid(*[np.linspace(bounds[p][0], bounds[p][1], nside)
                           for p in pars])
    X = np.vstack([p.reshape(-1,) for p in points]).T
else:
    X = utils.load_pickle('./data/submit_points/AM_{}_{}_{}_{}_{}.p'.format(
        args.name, args.file_index, args.proxy, args.scope[0], args.scope[1]))
    if rank == 0:
        print('Running with cached points')

#
#  -------------------------------------------------------------------------- #
#                   Split the job among MPI                                   #
# --------------------------------------------------------------------------- #
#


Njobs = X.shape[0]
N_per_proc = int(np.floor(Njobs / size))


def sample(index):
    point = {p: X[index, i] for i, p in enumerate(pars)}
    lp, blobs = model(point)
    data = {'point': point, 'lp': lp, 'blobs': blobs}
    fname = './results/{}/{}_AM_{}_{}_{}_{}.p'.format(
            args.name, index, args.proxy, args.file_index, args.scope[0],
            args.scope[1])
    utils.dump_pickle(fname, data)
    print('Finished {}'.format(index))


if rank != size - 1:
    for i in range(N_per_proc):
        sample(rank * N_per_proc + i)
else:
    for i in range(N_per_proc * rank, Njobs):
        sample(i)

#
#  -------------------------------------------------------------------------- #
#                       Collect outputs                                       #
# --------------------------------------------------------------------------- #
#

# This causes threads to wait until they've all finished
buff = np.zeros(1)
if rank == 0:
    for i in range(1, size):
        comm.Recv(buff, source=i)
else:
    comm.Send(buff, dest=0)

if rank == 0:
    file_out = {'point': [],
                'lp': [],
                'blobs': []}
    for i in range(Njobs):
        fname = './results/{}/{}_AM_{}_{}_{}_{}.p'.format(
                args.name, i, args.proxy, args.file_index, args.scope[0],
                args.scope[1])
        inp = utils.load_pickle(fname)
        for key in file_out.keys():
            file_out[key].append(inp[key])
        os.remove(fname)

    fname = './results/{}/AM_{}_{}_{}_{}.p'.format(
            args.name, args.proxy, args.file_index, args.scope[0],
            args.scope[1])
    utils.dump_pickle(fname, file_out)

    try:
        os.remove('./data/submit_points/AM_{}_{}_{}_{}_{}.p'.format(
            args.name, args.file_index, args.proxy, args.scope[0],
            args.scope[1]))
    except FileNotFoundError:
        pass

    print('All finished')
