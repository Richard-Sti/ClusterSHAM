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
    """Finish writing docs and setters for this... """

    def __init__(self, AM_generator, likelihood_model, cluster_model, bounds, Nmocks, cut_range):
        self._AM_generator = None
        self._likelihood_model = None
        self._prior_dist = None
        self._cluster_model = None
        self._bounds = None

        self.bounds = bounds
        self.Nmocks = Nmocks
        self.cut_range = cut_range

        self.AM_generator = AM_generator
        self.likelihood_model = likelihood_model
        self.cluster_model = cluster_model



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

    def mock_wps(self, theta, halos):
        """ Docs """
        deconv_cat = self.AM_generator.deconvoluted_catalogs(theta, halos)

        wps = numpy.zeros(shape=(self.Nmocks, self.cluster_model.Nrpbins))
        for i in range(self.Nmocks):
            mask, cat = self.AM_generator.add_scatter(deconv_cat,
                                                      self.cut_range)

            if i == 0:
                cov_jack, wp = self.cluster_model.mock_jackknife_cov(
                        halos['x'][mask], halos['y'][mask], halos['z'][mask],
                        return_wp=True)
            else:
                wp = self.cluster_model.mock_wp(
                        halos['x'][mask], halos['y'][mask], halos['z'][mask])

            wps[i, :] = wp

        cov_stoch = numpy.cov(wps, rowvar=False, bias=True)
        mean_wp = numpy.mean(wps, axis=0)

        return mean_wp, cov_stoch, cov_jack

    def __call__(self, theta, halos, return_blobs=False):

        logp = sum(self.prior_dist[key].logpdf(val)
                   for key, val in theta.items())
        if not numpy.isfinite(logp):
            if return_blobs:
                return numpy.nan, {}
            return numpy.nan

        mean_wp, cov_stoch, cov_jack = self.mock_wps(theta, halos)
        logl = self.likelihood_model(mean_wp, cov_stoch, cov_jack)

        if return_blobs:
            blobs = {'logl': logl,
                     'logp': logp,
                     'wp': mean_wp,
                     'cov_stoch': cov_stoch,
                     'cov_jack': cov_jack}
            return logp + logl, blobs

        return logp + logl

