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
"""A class that does abundance matching."""

import random

import numpy as np

from joblib import (Parallel, delayed, externals)
from AbundanceMatching import (AbundanceFunction, add_scatter, rematch,
                               calc_number_densities, LF_SCATTER_MULT)

from .base import BaseAbundanceMatch

MAX_INT = 2**32 - 1


class AbundanceMatch(BaseAbundanceMatch):

    r"""A wrapper around Yao-Yuan Mao's Subhalo Abundance Matching (SHAM)
    Python package [1]. Performs abundance matching on a list of halos.

    Parameters
    ----------
    nd_gal : numpy.ndarray
        Number density of galaxies to be used. This can be either a luminosity
        or a mass function. Shape must be (N, 2), where N is the number of
        bins. The first column must be the absolute magnitude or log mass
        and the second column the number density of that bin.
    scope : tuple
        Magnitude or log mass range over which to perform abundance matching.
        Must be a len-2 tuple.
    halos : numpy.ndarray
        Array containing the halos. ``halos.dtype.names`` must include the
        Cartesian coordinates and the halo proxy parameters.
    halo_proxy : PySHAM.mocks.proxies
        A halo proxy object.
    boxsize : int
        Length of a side of the simulation box in which halos are located.
    Nmocks : int
        Number of mocks to produce at a given posterior point.
    nthreads : int
        Number of threads.

    References
    ----------
    .. [1] https://bitbucket.org/yymao/abundancematching/

    """

    def __init__(self, nd_gal, scope, halos, boxsize, halo_proxy, Nmocks=1,
                 nthreads=1):
        self.nd_gal = nd_gal
        self.scope = scope
        self.halos = halos
        self.boxsize = boxsize
        self.halo_proxy = halo_proxy
        self.Nmocks = Nmocks
        self.nthreads = nthreads

        self._set_xrange(scope)

    def _set_xrange(self, scope):
        """Sets the xrange over which abundance matching is done."""
        dx = 2.5
        if self.is_luminosity:
            dx *= LF_SCATTER_MULT
        self._xrange = (scope[0] - dx, scope[1] + dx)

    def _seeds(self):
        """Returns an array of seeds, ensuring all are unique."""
        while True:
            random.seed()
            seeds = [random.randint(0, MAX_INT) for __ in range(self.Nmocks)]
            seeds = list(set(seeds))
            if len(seeds) == self.Nmocks:
                return seeds

    def match(self, theta):
        """Matches galaxies to halos."""
        scatter = theta.pop('scatter')
        plist = self.halo_proxy.proxy(self.halos, **theta)
        nd_halos = calc_number_densities(plist, self.boxsize)
        # LF scatter has another scalar multiplying it (2.5)
        if self.is_luminosity:
            scatter *= LF_SCATTER_MULT
        af = AbundanceFunction(self.nd_gal[:, 0], self.nd_gal[:, 1],
                               self._xrange,
                               faint_end_first=self._faint_end_first)
        # Deconvolute the scatter
        af.deconvolute(scatter, repeat=20)
        # Catalog with 0 scatter
        cat = af.match(nd_halos)
        cat_dec = af.match(nd_halos, scatter, False)
        # Start generating catalogs. Ensure different random seeds
        seeds = self._seeds()
        if self.nthreads > 1:
            with Parallel(self.nthreads, verbose=10, backend='loky') as par:
                masks = par(delayed(self._scatter_mask)(
                    i, cat, cat_dec, scatter, af._x_flipped) for i in seeds)
            # clean up the parallel pools
            externals.loky.get_reusable_executor().shutdown(wait=True)
        else:
            masks = [self._scatter_mask(i, cat, cat_dec, scatter,
                                        af._x_flipped) for i in seeds]
        return [[self.halos[p][mask] for p in ['x', 'y', 'y']]
                for mask in masks]

    def _scatter_mask(self, seed, cat, cat_dec, scatter, flipped):
        """Rematches galaxies and picks only ones in scope."""
        np.random.seed(seed)
        out = rematch(add_scatter(cat_dec, scatter), cat, flipped)
        # Eliminate NaNs and galaxies with mass/brightness below/above the cut
        x0, xf = self.scope
        return (~np.isnan(out)) & (x0 < out) & (out < xf)
