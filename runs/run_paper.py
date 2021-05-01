# Copyright (C) 2021  Richard Stiskalek
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

import sys
import argparse
from time import time
from datetime import datetime
import numpy

from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import joblib

from gaussian_search import GaussianProcessSearch
from parser_paper_model import PaperModelConfigParser


def time_evaluation(model, kwargs):
    """
    Prints out how long it takes to sample the posterior.
    """
    x = numpy.linspace(0.1, 1.5, 5)
    y = numpy.linspace(0.006, 0.3, 5)

    thetas = [{'alpha': xi, 'scatter': yi} for xi, yi in zip(x, y)]
    for theta in thetas:
        start = time()
        lp, blobs = model(theta.copy(), **kwargs)
        print(theta)
        sys.stdout.flush()
        print('Duration', time() - start)
        print(lp)
        sys.stdout.flush()


def time_evaluation_parallel(model, kwargs):
    """
    Prints out how long it takes to sample the posterior in parallel.
    """
    x = numpy.linspace(0.1, 1.5, 8)
    y = numpy.linspace(0.006, 0.3, 8)

    thetas = [{'alpha': xi, 'scatter': yi} for xi, yi in zip(x, y)]

    print("Initial evaluation")
    sys.stdout.flush()
    theta0 = {'alpha': 0.9, 'scatter': 0.13}
    start = time()
    model(theta0, nthreads=4, **kwargs)
    print("Initial evaluation complete in {} seconds.".format(time() - start))
    sys.stdout.flush()

    start = time()
    with joblib.Parallel(n_jobs=4) as par:
        par(joblib.delayed(model)(point, **kwargs) for point in thetas.copy())
    print('Duration for first 8', time() - start)
    sys.stdout.flush()

    start = time()
    with joblib.Parallel(n_jobs=4) as par:
        par(joblib.delayed(model)(point, **kwargs) for point in thetas.copy())
    print('Duration for next 8', time() - start)
    sys.stdout.flush()


def argument_parser():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description='Paper model submitted.')
    parser.add_argument('--path', type=str, help='Config file path.')
    parser.add_argument('--sub_id', type=str, help='Subsample ID.')
    parser.add_argument('--Ninit', type=int,
                        help='Number of initial uniform batches.')
    parser.add_argument('--Nmcmc', type=int, help='Number of MCMC batches.')
    parser.add_argument('--nthreads', type=int,
                        help="Number of threads, will also be the batch size.")
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path.',
                        default=None)
    parser.add_argument('--beta', type=float, default=1,
                        help='Acquisition parallel tempering inverse temp.')
    args = parser.parse_args()
    return args


def clean_grid(model, parser, args):
    """
    Initialises a clean grid search.
    """
    # Get the Gaussian process grid search
    name = "{}_{}".format(parser.cnf['Main']['attr'], args.sub_id)
    bounds = parser.cnf['Bounds']
    params = list(bounds.keys())
    sampler_kwargs = {'nlive': 1000}

    length_scale = [0.1 * abs(bounds[p][1] - bounds[p][0]) for p in params]
    kernel = kernels.Matern(length_scale=length_scale, nu=2.5)
    gp = GaussianProcessRegressor(kernel, alpha=1e-6)
    hyper_grid = {'gp__alpha': numpy.logspace(-4, -1, 25),
                  'gp__kernel__nu': [2.5, numpy.infty]}
    return GaussianProcessSearch(name, params, model, bounds,
                                 random_state=42, nthreads=args.nthreads,
                                 sampler_kwargs=sampler_kwargs, gp=gp,
                                 hyper_grid=hyper_grid)


def from_checkpoint(model, args):
    """
    Loads a grid search instance from a checkpoint file.
    """
    checkpoint = joblib.load(args.checkpoint)
    grid = GaussianProcessSearch.from_checkpoint(model, checkpoint)
    try:
        cache = joblib.load('./temp/correlator_RRpairs_{}.z'.format(grid.name))
        state = joblib.load('./temp/correlator_random_state_{}.z'
                            .format(grid.name))
        print("{}: Cached RR pairs found, loading.".format(datetime.now()))
        grid.logmodel.correlator._cache = cache
        grid.logmodel.correlator._random_state = state
    except FileNotFoundError:
        pass
    return grid


def main():
    args = argument_parser()
    print("{}: We are working {} for {} at inverse temperature {}."
          .format(datetime.now(), args.path, args.sub_id, args.beta))
    parser = PaperModelConfigParser(args.path, args.sub_id)
    model = parser()

    print("{}: Loading halos.".format(datetime.now()))
    sys.stdout.flush()
    kwargs = {'halos': numpy.load(parser.cnf['Main']['halos_path']),
              'return_blobs': True}
#    time_evaluation(model, kwargs)
#    time_evaluation_parallel(model, kwargs)
#    return

    if args.checkpoint is None:
        grid = clean_grid(model, parser, args)
    else:
        grid = from_checkpoint(model, args)
    # Ensure that the grid boundary correspond to the config file
    grid.bounds = model.bounds

    if grid.logmodel.correlator._cache == {}:
        print("{}: Pre-calculating the RR pairs.".format(datetime.now()))
        sys.stdout.flush()
        # Warm-up run to cache the data before splitting among parallel procs.
        p0 = {p: grid._uniform_samples(1)[0, i]
              for i, p in enumerate(grid.params)}
        grid.logmodel(p0, nthreads=args.nthreads, **kwargs)

        joblib.dump(grid.logmodel.correlator._cache,
                    './temp/correlator_RRpairs_{}.z'.format(grid.name))
        joblib.dump(grid.logmodel.correlator._random_state,
                    './temp/correlator_random_state_{}.z'.format(grid.name))
        print("{}: Dumped the cached RR pairs.".format(datetime.now()))

    print("{}: Starting the grid search.".format(datetime.now()))
    sys.stdout.flush()
    grid.run_batches(Ninit=args.Ninit, Nmcmc=args.Nmcmc,
                     batch_size=5*args.nthreads, kwargs=kwargs, beta=args.beta)


if __name__ == '__main__':
    main()
