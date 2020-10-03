#!/usr/bin/env python
# coding: utf-8
import numpy as np
from AbundanceMatching import (AbundanceFunction, add_scatter, rematch,
                               calc_number_densities, LF_SCATTER_MULT)
from joblib import (Parallel, delayed, externals)
import random


class AbundanceMatch(object):
    def __init__(self, nd_gal, halos, scope, proxy_func, Njobs=8, Nmocks=48,
                 boxsize=400):
        self.halos = halos
        self.nd_gal = nd_gal
        self.scope = scope
        self.boxsize = boxsize
        self.Nmocks = Nmocks
        self.Njobs = Njobs
        self.proxy_func = proxy_func

        if self._AM_type == 'MF':
            self._xrange = (8.0, 13.5)
            self._faint = True
        else:
            self._xrange = (-27.0, -16.5)
            self._faint = False

    @property
    def _AM_type(self):
        if self.scope[0] > 0:
            return 'MF'
        else:
            return 'LF'

    def abundance_match(self, theta):
        scatter = theta['scatter']
        plist = self.proxy_func(self.halos, **theta)
        nd_halos = calc_number_densities(plist, self.boxsize)

        if self._AM_type == 'LF':
            scatter *= LF_SCATTER_MULT
        af = AbundanceFunction(self.nd_gal[:, 0], self.nd_gal[:, 1],
                               self._xrange, faint_end_first=self._faint)
        # Deconvolute the scatter
        af.deconvolute(scatter, repeat=20)
        # Catalog with 0 scatter
        cat = af.match(nd_halos)
        cat_dec = af.match(nd_halos, scatter, False)
        # Start generating catalogs. Ensure different random seeds
        while True:
            random.seed()
            seeds = np.random.randint(0, 2**32 - 1, size=self.Nmocks)
            __, counts = np.unique(seeds, return_counts=True)
            if not np.any(counts > 1):
                break

        if self.Njobs > 1:
            with Parallel(self.Njobs, verbose=10, backend='loky') as par:
                masks = par(delayed(self._scatter_mask)(seed, cat, cat_dec,
                                                        scatter,
                                                        af._x_flipped)
                            for seed in seeds)
            # clean up the parallel pools
            externals.loky.get_reusable_executor().shutdown(wait=True)
        else:
            masks = [self._scatter_mask(seed, cat, cat_dec, scatter,
                                        af._x_flipped) for seed in seeds]

        samples = [(self.halos['x'][mask], self.halos['y'][mask],
                   self.halos['z'][mask]) for mask in masks]
        return samples

    def _scatter_mask(self, seed, cat, cat_dec, scatter, flipped):
        np.random.seed(seed)
        out = rematch(add_scatter(cat_dec, scatter), cat, flipped)
        # Eliminate NaNs and galaxies with mass/brightness below the cut
        x0, xf = self.scope
        mask = (~np.isnan(out)) & (x0 < out) & (out < xf)
        return mask
