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
"""Likelihood models to contrain the galaxy-halo connection."""

from random import choice

import numpy as np
from scipy.stats import multivariate_normal

import Corrfunc

from .jackknife import Jackknife
from .base import BaseClusteringLikelihood


class ClusteringLikelihood(BaseClusteringLikelihood):

    r""" A clustering likelihood on the projected two-point correlation
    functions. Assumes a normal distribution on the residuals.

    Parameters
    ----------
    parameters : (a list of) str
        Parameter names for the likelihood.
    wp_survey : numpy.ndarray
        Survey correlation function. Must be calculated over bins specified
        in ``rpbins``.
    cov_survey : numpy.ndarray
        Survey correlation function covariance matrix. Must be calculated over
        bins specified in ``rpbins``.
    AM_model : PySHAM.mocks.AbundaceMatching
        The abundance matching model object as specified in PySHAM.
    subside : int
        Lenght of a subvolume being removed at each turned. This subvolume
        is assumed to be ``subside`` x ``subside`` x ``boxsize.
    rpbins : numpy.ndarray
        Array of bins orthogonal to the line of sight in which the
        project correlation function is calculated.
    pimax : float
        Maximum distance along the line of sight to integrate over when
        calculating the projected wp.
    nthreads : int, optional
        Number of threads.
    """

    def __init__(self, parameters, wp_survey, cov_survey, AM_model, subside,
                 rpbins, pimax, nthreads=1):
        # parse inputs
        self.parameters = parameters
        self.wp_survey = wp_survey
        self.cov_survey = cov_survey
        self.AM_model = AM_model
        self.rpbins = rpbins
        self.pimax = pimax
        self.nthreads = nthreads
        # define the jackknife model
        self.jackknife_model = Jackknife(subside=subside, rpbins=rpbins,
                                         boxsize=AM_model.boxsize,
                                         pimax=pimax, nthreads=nthreads)

    def stochastic_variation(self, samples):
        """Calculates the mean and covariance matrix for a list of catalogs.
        Assumes catalogs are independent for 1/N normalisation.
        """
        wps = [None] * len(samples)
        for i, sample in enumerate(samples):
            X, Y, Z = sample
            out = Corrfunc.theory.wp(boxsize=self.AM_model.boxsize,
                                     pimax=self.pimax, nthreads=self.nthreads,
                                     binfile=self.rpbins, X=X, Y=Y, Z=Z)['wp']
            wps[i] = out
        wps = np.array(wps)
        # Bias=True means normalisation by 1/N instead of 1/(N-1)
        return np.mean(wps, axis=0), np.cov(wps, rowvar=False, bias=True)

    def logpdf(self, theta, return_blobs=True):
        samples = self.AM_model.match(theta)
        wp_stoch, cov_stoch = self.stochastic_variation(samples)
        # Randomly pick one AM mock to jackknife
        cov_jack = self.jackknife_model.jackknife(choice(samples))
        # All covariances are added
        cov = cov_stoch + cov_jack + self.cov_survey
        ll = multivariate_normal.logpdf(wp_stoch, mean=self.wp_survey, cov=cov)
        if return_blobs:
            blobs = {'wp': wp_stoch, 'cov_jack': cov_jack,
                     'cov_stoch': cov_stoch}
            return ll, blobs
        return ll
