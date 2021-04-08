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

import numpy
from scipy.stats import uniform


class PaperModel:
    """
    The clustering-calibrated subhalo abundance matching model of [1].
    Compares the mock galaxy clustering (determined by the halo proxy) to
    observations. Assumes the standard Gaussian likelihood and uniform priors
    on halo proxy parameters and scatter.

    At each halo proxy parametrisation generates 50 mocks (with a different
    seed), calculates the jackknife covariance matrix using the first mock,
    and calculates the 2-point correlation function for all remaining mocks.

    Parameters
    ----------
    AM : `clustersham.mocks.AbundanceMatch` object
        Abundance matching interface to generate the mock galaxy catalogs.
    correlator : `clustersham.mocks.Correlator` object
        Correlator object to calculate the 2-point statistics.
    likelihood : `clustersham.utils.GaussianClusteringLikelihood` object
        Likelihood object to calculate the log-probability.
    bounds : dict
        Dictionary with parameter boundaries {key: (min, max)}.
    cut_range : len-2 tuple
        Galaxy proxy range for which to calculate the statistics.
    Nmocks : int, optional
        Number of galaxy mocks to generate at a fixed halo proxy and scatter.
        By default 50.
    seed : int, optional
        Random seed.


    References
    ----------
    .. [1] https://arxiv.org/abs/2101.02765
    """

    def __init__(self, AM, correlator, likelihood, bounds, cut_range,
                 Nmocks=50, seed=42):
        self._AM = None
        self._correlator = None
        self._likelihood = None
        self._bounds = None
        self._prior_dist = None
        self._Nmocks = None
        self._cut_range = None

        self.AM = AM
        self.correlator = correlator
        self.likelihood = likelihood
        self.bounds = bounds
        self.Nmocks = Nmocks
        self.cut_range = cut_range
        # Set the random seed
        numpy.random.seed(seed)

    @property
    def AM(self):
        """The abundance matching generator."""
        return self._AM

    @AM.setter
    def AM(self, AM):
        """Sets `AM`."""
        err = False
        try:
            if AM.name != "AbundanceMatch":
                err = True
        except AttributeError:
            err = True
        if err:
            raise ValueError("`AM` must be of "
                             "`clustersham.mocks.AbundanceMatch` type.")
        self._AM = AM

    @property
    def correlator(self):
        """The correlator. Used to count the 2-point statistics."""
        return self._correlator

    @correlator.setter
    def correlator(self, correlator):
        """Sets `correlator`."""
        err = False
        try:
            if correlator.name != "Correlator":
                err = True
        except AttributeError:
            err = True
        if err:
            raise ValueError("`correlator` must be of "
                             "`clustersham.mocks.Correlator` type.")
        self._correlator = correlator

    @property
    def likelihood(self):
        """The likelihood object."""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        """Sets `likelihood`."""
        err = False
        try:
            if likelihood.name != "GaussianClusteringLikelihood":
                err = True
        except AttributeError:
            err = True
        if err:
            raise ValueError(
                    "`likelihood` must be of "
                    "`clustersham.utils.GaussianClusteringLikelihood` type.")
        self._likelihood = likelihood

    @property
    def bounds(self):
        """Prior bounds on each parameter."""
        return self._bounds

    @bounds.setter
    def bounds(self, bnds0):
        """
        Sets `bounds`. Checks everything and initialises the prior
        distributions.
        """
        bnds = {}
        prior_dist = {}
        if not isinstance(bnds0, dict):
            raise ValueError("`bounds` must be of dict type.")
        for key, bnd in bnds0.items():
            if len(bnd) != 2:
                raise ValueError("Bound on '{}' must be length 2."
                                 .format(key))
            if bnd[0] > bnd[1]:
                bnd = bnd[::-1]

            prior_dist.update({key: uniform(bnd[0], abs(bnd[1] - bnd[0]))})
            bnds.update({key: bnd})

        self._bounds = bnds
        self._prior_dist = prior_dist

    @property
    def prior_dist(self):
        """Prior uniform distribution on each parameter."""
        return self._prior_dist

    @property
    def Nmocks(self):
        """Number of mock catalogues to be generated at point."""
        return self._Nmocks

    @Nmocks.setter
    def Nmocks(self, N):
        """Sets `Nmocks`."""
        if not isinstance(N, int):
            raise ValueError("`Nmocks` must be an integer.")
        self._Nmocks = N

    @property
    def cut_range(self):
        """Galaxy proxy range for which to calculate the statistics."""
        return self._cut_range

    @cut_range.setter
    def cut_range(self, cut_range):
        """Sets `cut_range`."""
        if len(cut_range) != 2:
            raise ValueError("`cut_range` must be a tuple of len 2.")
        if cut_range[0] > cut_range[1]:
            cut_range = cut_range[::-1]
        self._cut_range = cut_range

    def mock_wps(self, theta, halos):
        """
        Iteratively generates `self.Nmocks` galaxy mocks. At each iteration
        the 2-point correlation function is calculated. Jackknifes the
        simulation box on the first iteration.

        Parameters
        ----------
        theta : dict
            Dictionary of halo proxy parameters and scatter.
        halos : structured numpy.ndarray
            Array (with named fields) with halo properties.

        Returns
        -------
        mean_wp : numpy.ndarray
            The mean 2-point correlation function.
        cov_stoch : numpy.ndarray
            The stochastic covariance matrix.
        cov_jack : numpy.ndarray
            The jackknife covariance matrix.
        """
        deconv_cat = self.AM.deconvoluted_catalogs(theta, halos)

        wps = numpy.zeros(shape=(self.Nmocks, self.correlator.Nrpbins))
        for i in range(self.Nmocks):
            mask = self.AM.add_scatter(deconv_cat, self.cut_range)

            if i == 0:
                cov_jack, wp = self.correlator.mock_jackknife_cov(
                        halos['x'][mask], halos['y'][mask], halos['z'][mask],
                        return_wp=True)
            else:
                wp = self.correlator.mock_wp(
                        halos['x'][mask], halos['y'][mask], halos['z'][mask])

            wps[i, :] = wp

        cov_stoch = numpy.cov(wps, rowvar=False, bias=True)
        mean_wp = numpy.mean(wps, axis=0)

        return mean_wp, cov_stoch, cov_jack

    def __call__(self, theta, halos, return_blobs=False):
        """
        Calculates the log-posterior for halo proxy parameters and scatter.
        Returns `numpy.nan` if outside of prior boundaries.

        Parameters
        ----------
        theta : dict
            Dictionary of halo proxy parameters and scatter.
        halos : structured numpy.ndarray
            Array (with named fields) with halo properties.
        return_blobs : bool, optional
            Whether to return blobs (logprior, loglikelihood, the mean
            2-point correlation function and the covariance matrices).
            By default False.

        Returns
        -------
        logposterior : float
            The log posterior of the halo proxy parameters and scatter.
        blobs : dict
            Returned if `return_blobs`. If outside of prior boundaries returns
            an empty dictionary, otherwise contains:
                log_l : float
                    The log likelihood.
                log_p : float
                    The log prior.
                wp : numpy.ndarray
                    The mean 2-point correlation function.
                cov_stoch : numpy.ndarray
                    The stochastic covariance matrix.
                cov_jack : numpy.ndarray
                    The jackknife covariance matrix.
        """
        logp = sum(self.prior_dist[key].logpdf(val)
                   for key, val in theta.items())
        if not numpy.isfinite(logp):
            if return_blobs:
                return numpy.nan, {}
            return numpy.nan

        mean_wp, cov_stoch, cov_jack = self.mock_wps(theta, halos)
        logl = self.likelihood(mean_wp, cov_stoch, cov_jack)

        if return_blobs:
            blobs = {'logl': logl,
                     'logp': logp,
                     'wp': mean_wp,
                     'cov_stoch': cov_stoch,
                     'cov_jack': cov_jack}
            return logp + logl, blobs

        return logp + logl
