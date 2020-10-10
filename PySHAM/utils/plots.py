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
"""Classes for making plots."""

import numpy as np

from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import uniform

from . import in_hull


class BasePlots(object):
    r"""A base class for the the Plots class. Handles inputs and returning
    values.

    Attributes
    ----------
    parameters
    rpbins
    sampled_points
    wp_survey
    cov_survey
    boundaries
    """
    _parameters = None
    _rpbins = None
    _sampled_points = None
    _wp_survey = None
    _cov_survey = None
    _boundaries = None

    @property
    def parameters(self):
        """Returns the parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, pars):
        """Sets the parameters."""
        if isinstance(pars, str):
            pars = [pars]
        if not isinstance(pars, (list, tuple)):
            raise ValueError("``parameters`` must be specified as a list.")
        pars = list(pars)
        if not all(isinstance(p, str) for p in pars):
            raise ValueError("All ``parameters`` must be str.")
        self._parameters = pars

    @property
    def rpbins(self):
        """Returns the rp bins which the correlation functions were calculated.
        """
        return self._rpbins

    @rpbins.setter
    def rpbins(self, rpbins):
        """Sets the rp bins."""
        if not isinstance(rpbins, np.ndarray):
            raise ValueError("``rpbins`` must be of numpy.ndarray type.")
        self._rpbins = rpbins

    @property
    def x(self):
        """Returns the centers of rpbins."""
        bins = self.rpbins
        return [0.5 * (bins[i+1] + bins[i]) for i in range(len(bins))]

    @property
    def sampled_points(self):
        """Returns a list of sampled points. The list's items are dictionaries
        of the sampled point, logprior, loglikelihood, mean simulated
        correlation function and the stochastic and jackknife covariance
        matrix estimates.
        """
        return self._sampled_points

    @sampled_points.setter
    def sampled_points(self, blobs):
        """Sets the sampled points."""
        if not isinstance(blobs, list):
            raise ValueError("``blobs`` must be of list type.")
        for i, blob in enumerate(blobs):
            if not all(p in blob.keys() for p in ['theta', 'loglikelihood',
                                                  'logprior', 'wp', 'cov_jack',
                                                  'cov_stoch']):
                raise ValueError("{}th blob is not valid.".format(i))
        self._sampled_points = blobs

    @property
    def stats(self):
        """Returns the sampled positions and stats as a structured numpy
        array.
        """
        stats_pars = ['logprior', 'loglikelihood']
        pars = list(self.sampled_points[0]['theta'].keys()) + stats_pars
        formats = ['float64'] * len(pars)
        N = len(self.sampled_points)
        out = np.zeros(N, dtype={'names': pars, 'formats': formats})
        for p in pars:
            if p in stats_pars:
                out[p] = [point[p] for point in self.sampled_points]
            else:
                out[p] = [point['theta'][p] for point in self.sampled_points]
        return out

    @property
    def wp_simulation(self):
        """Returns the simulated correlation functions in a list whose order
        corresponds to ``self.points``.
        """
        return [point['wp'] for point in self.sampled_points]

    @property
    def cov_simulation(self):
        """Returns the covariance matrices for the simulated correlation
        functions in a list whose order corresponds to ``self.wp_simulation``.
        """
        return [p['cov_stoch'] + p['cov_jack'] for p in self.sampled_points]

    @property
    def wp_survey(self):
        """Returns the survey correlation function corresponding to this bin.
        """
        return self._wp_survey

    @wp_survey.setter
    def wp_survey(self, wp):
        if not isinstance(wp, np.ndarray):
            raise ValueError("``wp_survey`` must be of numpy.ndarray type.")
        self._wp_survey = wp

    @property
    def cov_survey(self):
        """Returns the survey covariance matrix corresponding to
        ``self.wp_survey``."""
        return self._cov_survey

    @cov_survey.setter
    def cov_survey(self, cov):
        if not isinstance(cov, np.ndarray):
            raise ValueError("``cov_survey`` must be of numpy.ndarray type.")
        self._cov_survey = cov

    @property
    def boundaries(self):
        """Prior boundaries on posterior parameters."""
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries):
        """Sets the boundaries."""
        if not isinstance(boundaries, dict):
            raise ValueError("``boundaries`` must be a dictionary.")
        for key in boundaries.keys():
            if key not in self.parameters:
                raise ValueError("Missing boundaries for {}".format(key))
        self._boundaries = boundaries


class Plots(BasePlots):
    r"""A class for simple plot making. Specifically handles contour plots,
    evidence calculation and best-fit plots.

    Parameters
    ----------
    parameters: (list of) str
        Posterior parameter names.
    blobs: list
        Values returned by the posterior. Each item must be a dictionary and
        contain ``theta``, ``loglikelihood``, ``logprior``, ``wp``,
        ``cov_stoch``, ``cov_jack``.
    wp_survey: numpy.ndarray
        Correlation function of the survey corresponding to the scope of
        abundance matching.
    wp_cov: numpy.ndarray
        Covariance matrix for ``wp_survey``.
    rp_bins: numpy.ndarrya
        Bins used to calculate the projected correlation function.
    prior_boundaries: dict
        A dictionary of parameter names -> boundaries. Boundaries must be an
        increasing tuple of length 2.
    """

    def __init__(self, parameters, blobs, wp_survey, wp_cov, rpbins,
                 prior_boundaries):
        self.parameters = parameters
        self.sampled_points = blobs
        self.wp_survey = wp_survey
        self.wp_cov = wp_cov
        self.rpbins = rpbins
        self.boundaries = prior_boundaries

    def _padding(self, nside):
        """
        L
        """
        axes = [None] * len(self.parameters)
        for i, p in enumerate(self.parameters):
            a, b = self.boundaries[p]
            axes[i] = np.linspace(a, b, nside)

        Xgrid = np.vstack([axis.reshape(-1,) for axis in np.meshgrid(*axes)]).T
        # create a hull with sampled points
        Xsampled = np.vstack([self.stats[par] for par in self.parameters]).T
        hull = ConvexHull(Xsampled)
        # which of the grid points are inside the hull
        Ngrid = Xgrid.shape[0]
        mask = np.array([in_hull(Xgrid[i, :], hull) for i in range(Ngrid)])
        # pad the sampled points
        Xinterpolator = np.vstack([Xsampled, Xgrid[np.logical_not(mask)]])
        return Xinterpolator, Xgrid

    def contour_2D(self, nside, smooth_std=None):
        """
        L

        """
        Xinterp, Xgrid = self._padding(nside)
        # make padding loglikelihood just the sampled minimum
        Zinterp = np.full(Xinterp.shape[0], self.stats['loglikelihood'].min())
        Zinterp[:len(self.sampled_points)] = self.stats['loglikelihood']
        # this linear interpolator will evaluate the regular grid points
        f = LinearNDInterpolator(Xinterp, Zinterp, rescale=True)
        Zgrid = f(Xgrid)
        # possibly add smoothing to make nice contours
        if smooth_std is not None:
            Zgrid = gaussian_filter(Zgrid, smooth_std)
        # change nans to -np.infty
        contours = self._sigma_regions(Zgrid)
        # reshape for matplotlib.pyplot.contour
        Xgrid = Xgrid.reshape(nside, nside, 2)
        Zgrid = Zgrid.reshape(nside, nside)
        args = [Xgrid[:, :, 0], Xgrid[:, :, 1], Zgrid, contours]
        return args

    @staticmethod
    def _sigma_regions(logZ):
        """
        L
        """
        logZ = np.ravel(logZ)
        prob = np.exp(logZ) / np.sum(np.exp(logZ))

        # sigma volume
        sigmas = [np.exp(-0.5 * x**2) for x in range(1, 3 + 1)][::-1]
        contours = [None] * 3

        IDS = np.argsort(prob)
        for i, sigma in enumerate(sigmas):
            # at what index in the ordered array do we hit the contour level
            index = np.abs(np.cumsum(prob[IDS]) - sigma).argmin()
            contours[i] = logZ[IDS[index]]
        return contours

    def uniform_prior(self, X):
        """
        L.
        """
        lp = np.zeros(X.shape[0])
        for i, p in enumerate(self.parameters):
            a, b = self.boundaries[p]
            lp += uniform.logpdf(x=X[:, i], loc=a, scale=(a + b))
        return lp

    def log_evidence_2D(self, nside):
        """
        L.
        """
        # padding around the sampled points
        Xinterp, Xgrid = self._padding(nside)
        Zinterp = np.zeros(Xinterp.shape[0])
        # for sampled points add loglikelihood
        Zinterp[:len(self.sampled_points)] += self.stats['loglikelihood']
#        Zinterp[len(self.sampled_points):] = -np.infty
        # for all points add logprior (uniform)
        for i, p in enumerate(self.parameters):
            a, b = self.boundaries[p]
            Zinterp += uniform.logpdf(Xinterp[:, i], loc=a, scale=a + b)
        # this linear interpolator will evaluate the regular grid points
        f = LinearNDInterpolator(Xinterp, Zinterp, rescale=True)
        Zgrid = f(Xgrid)
        Zgrid[np.isnan(Zgrid)] = -np.infty
        return Xgrid, Zgrid, np.log(np.sum(np.exp(Zgrid)))
#
#    def marginal_2D(self, par, nside=50):
#        """Sets a Gaussian process over the sampled points in 3D, evaluates
#        this on a fixed grid and subsequently marginalises over `par`"""
#        if not len(self.pars) == 3:
#            raise NotImplementedError('Only 3D posteriors supported')
#        indxs = [0, 1, 2]
#        ind_marg = self.pars.index(par)
#        indxs.pop(ind_marg)
#        ind1, ind2 = indxs[0], indxs[1]
#        # define the grid
#        grid = self._grid_points(nside)
#        gp = self._gp()
#        Z = gp.predict(grid)
#
#        X, Y = np.meshgrid(np.unique(grid[:, ind1]),
#                           np.unique(grid[:, ind2]))
#        Zmarg = np.zeros_like(X)
#        for i in range(nside):
#            for j in range(nside):
#                mask = np.intersect1d(np.where(grid[:, ind1] == X[i, j]),
#                                      np.where(grid[:, ind2] == Y[i, j]))
#                Zmarg[i, j] = sum(Z[mask])
#        return X, Y, Zmarg
