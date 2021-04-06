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
        return [0.5 * (bins[i+1] + bins[i]) for i in range(len(bins) - 1)]

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
        for par in pars:
            if par == 'loglikelihood':
                out[par] = [point[par] for point in self.sampled_points]
            elif par == 'logprior':
                # Calculate the logprior for each point
                lp = [0.0] * N
                bnd = self.boundaries
                dx = {p: abs(bnd[p][0] - bnd[p][1]) for p in self.parameters}
                x0 = {p: bnd[p][0] for p in self.parameters}
                for i, point in enumerate(self.sampled_points):
                    x = point['theta']
                    for p in self.parameters:
                        lp[i] += uniform.logpdf(x=x[p], loc=x0[p], scale=dx[p])
                out[par] = lp
            else:
                out[par] = [point['theta'][par]
                            for point in self.sampled_points]
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
    cov_survey: numpy.ndarray
        Covariance matrix for ``wp_survey``.
    rp_bins: numpy.ndarrya
        Bins used to calculate the projected correlation function.
    prior_boundaries: dict
        A dictionary of parameter names -> boundaries. Boundaries must be an
        increasing tuple of length 2.
    """

    def __init__(self, parameters, blobs, wp_survey, cov_survey, rpbins,
                 prior_boundaries):
        self.parameters = parameters
        self.sampled_points = blobs
        self.wp_survey = wp_survey
        self.cov_survey = cov_survey
        self.rpbins = rpbins
        self.boundaries = prior_boundaries

    def _uniform_grid(self, nside):
        """
        Returns uniformly spaced points within the prior boundaries.
        The total number of points is ``nside`` times ``nsides``.
        """
        axes = [None] * len(self.parameters)
        dx = [None] * len(self.parameters)
        for i, p in enumerate(self.parameters):
            a, b = self.boundaries[p]
            x = np.linspace(a, b, nside)
            dx[i] = x[1] - x[0]
            axes[i] = x
        Xgrid = np.vstack([axis.reshape(-1,) for axis in np.meshgrid(*axes)]).T
        return Xgrid, dx

    def contour_2D(self, nside, smooth_std=None, return_prior=False):
        """
        Returns the necessary arguments for a 2D contour plot: X, Y, logpost,
        and the contour levels.
        """
        X = np.vstack([self.stats[par] for par in self.parameters]).T
        logpost = self.stats['loglikelihood']
        if return_prior:
            logpost += self.stats['logprior']
        # this linear interpolator will be used to evaluate the grid points
        f = LinearNDInterpolator(X, logpost, rescale=True)
        # evaluate the interpolator on a grid
        Xgrid, __ = self._uniform_grid(nside)
        logpost = f(Xgrid)
        # possibly add smoothing to make nice contours
        if smooth_std is not None:
            logpost = gaussian_filter(logpost, smooth_std)
        contours = self._sigma_regions(logpost)
        # reshape for matplotlib.pyplot.contour
        Xgrid = Xgrid.reshape(nside, nside, 2)
        logpost = logpost.reshape(nside, nside)
        return Xgrid[:, :, 0], Xgrid[:, :, 1], logpost, contours

    def MCMC_samples_from_array(self, N, X, Y, logpost, bounds=None):
        """Returns ``N`` posterior samples from a 2D log posterior defined by
        ``logpost`` on a meshgrid of ``X`` and ``Y``.
        """
        # Appropriately normalise the log posterior
        post = np.exp(logpost)
        post /= np.max(post)
        post = post.reshape(-1, 1)
        features = np.array([X.reshape(-1, ), Y.reshape(-1, )]).T
        survival = LinearNDInterpolator(features, post, rescale=True)
        # Get the point generator
        if bounds is None:
            bnds = self.boundaries
        else:
            bnds = bounds
        loc = [bnds[p][0] for p in self.parameters]
        scale = [abs(bnds[p][1] - bnds[p][0]) for p in self.parameters]
        generator = uniform(loc=loc, scale=scale)

        MCMC_points = [None] * N
        i = 0
        while True:
            point = generator.rvs()
            if survival(point) > np.random.rand():
                MCMC_points[i] = point.tolist()
                i += 1
            if i == N:
                return MCMC_points

    @staticmethod
    def _sigma_regions(logZ):
        """
        Calculates the contours for ``logZ`` for the first three sigma levels
        on a 2D distribution.
        """
        logZ = np.ravel(logZ)
        prob = np.exp(logZ) / np.sum(np.exp(logZ))
        ncontours = 4

        # sigma volume
        sigmas = [np.exp(-0.5 * x**2) for x in range(1, ncontours + 1)][::-1]
        contours = [None] * ncontours

        IDS = np.argsort(prob)
        for i, sigma in enumerate(sigmas):
            # at what index in the ordered array do we hit the contour level
            index = np.abs(np.cumsum(prob[IDS]) - sigma).argmin()
            contours[i] = logZ[IDS[index]]
        return contours

    def _log_evidence_2D(self, nside):
        """
        Calculate the log evidence for an array defined by X, Y, and logZ,
        where logZ is the log posterior.
        """
        X = np.vstack([self.stats[par] for par in self.parameters]).T
        logpost = self.stats['loglikelihood'] + self.stats['logprior']
        # this linear interpolator will be used to evaluate the grid points
        f = LinearNDInterpolator(X, logpost, rescale=True)
        # evaluate the interpolator on a grid
        Xgrid, dX = self._uniform_grid(nside)
        logposterior = f(Xgrid)
        evidence = np.sum(np.exp(logposterior))
        return np.log(evidence) + sum(np.log(dx) for dx in dX)

    @staticmethod
    def log_evidence_from_array(X, Y, logposterior):
        """
        Returns the log evidence for ``logposterior`` specified on a uniform
        grid of ``X`` and ``Y``.
        """
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        evidence = np.sum(np.exp(logposterior))
        return np.log(evidence) + np.log(dx) + np.log(dy)

    def best_fit(self, nside=100):
        """Returns the best fit posterior point."""
        logposts = self.stats['loglikelihood'] + self.stats['logprior']
        i = np.argmax(logposts)
        point = self.sampled_points[i]

        AM_yerr = np.sqrt(np.diagonal(point['cov_stoch'] + point['cov_jack']))
        surv_yerr = np.sqrt(np.diagonal(self.cov_survey))

        AM_kwargs = {'x': self.x, 'y': point['wp'], 'yerr': AM_yerr}
        surv_kwargs = {'x': self.x, 'y': self.wp_survey, 'yerr': surv_yerr}
        fit_kwargs = {p: point['theta'][p] for p in self.parameters}
        fit_kwargs.update({'logposterior': logposts[i],
                           'logevidence': self._log_evidence_2D(nside)})
        return AM_kwargs, surv_kwargs, fit_kwargs

    def hull_randoms(self, npoints, vertices):
        """
        Generates ``npoints`` uniformly spaced points withing a convex hull
        specified by ``vertices.``
        """
        if isinstance(vertices, list):
            vertices = np.array(vertices)
        if not isinstance(vertices, np.ndarray):
            raise ValueError("``vertices`` must be a numpy.ndarray or a list")
        # create the hull
        hull = ConvexHull(vertices)
        points = list()
        # rejection sampling to get uniformly spaced points
        bnds = self.boundaries
        while len(points) < npoints:
            theta = [uniform.rvs(bnds[p][0], bnds[p][0] + bnds[p][1])
                     for p in self.parameters]
            if in_hull(theta, hull):
                points.append(theta)
        # convert into a numpy array
        points = np.array(points)
        return points



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
