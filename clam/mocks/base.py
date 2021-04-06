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
"""Base classes that handle inputs for the mocks submodule."""

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np


@add_metaclass(ABCMeta)
class Base(object):

    r"""Abstract class for handling inputs shared by most classes.

    Attributes
    ----------
    boxsize
    rpbins
    pimax
    nthreads
    """

    _boxsize = None
    _rpbins = None
    _pimax = None
    _nthreads = None

    @property
    def boxsize(self):
        """Length of a simulation box side."""
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """Sets the simulation box side."""
        if not isinstance(boxsize, (float, int)):
            raise ValueError("Must provide a float or int for ``boxsize``.")
        self._boxsize = float(boxsize)

    @property
    def rpbins(self):
        """Returns the r_p bins."""
        return self._rpbins

    @rpbins.setter
    def rpbins(self, rpbins):
        """Sets the rp bins."""
        if not isinstance(rpbins, np.ndarray):
            raise ValueError("``rpbins`` must be a numpy.ndarray type.")
        if rpbins.ndim != 1:
            raise ValueError("``rpbins`` must be a 1-dimensional array.")
        self._rpbins = rpbins

    @property
    def pimax(self):
        """Returns pimax, the maximum integration distance along the line of
        sight.
        """
        return self._pimax

    @pimax.setter
    def pimax(self, pimax):
        """Sets pimax."""
        if not isinstance(pimax, (int, float)):
            raise ValueError("``pimax`` must be int.")
        self._pimax = int(pimax)

    @property
    def nthreads(self):
        """Returns the number of threads."""
        return self._nthreads

    @nthreads.setter
    def nthreads(self, nthreads):
        """Sets the number of threads."""
        if nthreads < 1:
            raise ValueError("``nthreads`` must be larger than 1.")
        self._nthreads = int(nthreads)


@add_metaclass(ABCMeta)
class BaseClusteringLikelihood(Base):

    r""" Abstract class for handling inputs for the two-point projected
    correlation function clustering likelihood.

    Parameters
    ----------

    """
    _wp_survey = None
    _cov_survey = None
    _AM_model = None
    _jackknife_model = None

    @abstractmethod
    def logpdf(self, theta, return_blobs=True):
        """Returns the log probability density for parameters specified by
        theta.
        If ``return_blobs``, then returns the AM correlation function and
        covariance matrices.

        Parameters
        ----------
        theta : dict
            Dictionary of parameters to values.
        """
        pass

    @property
    def wp_survey(self):
        """Returns the survey 2-point correlation function calculated in bins
        which should match ``self.rpbins``."""
        return self._wp_survey

    @wp_survey.setter
    def wp_survey(self, wp):
        """Sets the survey wp."""
        if not isinstance(wp, np.ndarray):
            raise ValueError("``wp_survey`` must be of numpy.ndarray type.")
        if wp.ndim != 1:
            raise ValueError("``wp_survey`` must be a 1-d array.")
        self._wp_survey = wp

    @property
    def cov_survey(self):
        """Returns the survey covariance matrix estimate for
        ``self.wp_survey``."""
        return self._cov_survey

    @cov_survey.setter
    def cov_survey(self, cov):
        """Sets the survey covariance matrix."""
        if not isinstance(cov, np.ndarray):
            raise ValueError("``cov_survey`` must be of numpy.ndarray type.")
        if cov.ndim != 2:
            raise ValueError("``cov_survey`` must be a 2-d array.")
        self._cov_survey = cov

    @property
    def AM_model(self):
        """Returns the abundance matching model."""
        return self._AM_model

    @AM_model.setter
    def AM_model(self, AM_model):
        """Sets the AM model."""
        self._AM_model = AM_model

    @property
    def jackknife_model(self):
        """Returns the jackknifing model."""
        return self._jackknife_model

    @jackknife_model.setter
    def jackknife_model(self, jackknife_model):
        """Sets the jackknifing model."""
        self._jackknife_model = jackknife_model

    def __call__(self, theta):
        """Calls the loglikelihood and returns blobs."""
        return self.logpdf(theta)


@add_metaclass(ABCMeta)
class BaseProxy(object):
    r""" Abstract class for handling the abundance matching proxies. All
    proxies must inherit from this.

    Parameters
    ----------
    halos_parameters: (list of) str
        Names of halo parameters (properties) used to calculate the proxy.
    """
    _halos_parameters = None

    @property
    def halos_parameters(self):
        """Returns the halo parameters needed for the proxy calculation."""
        return self._halos_parameters

    @halos_parameters.setter
    def halos_parameters(self, pars):
        """Sets the halo parameters."""
        if isinstance(pars, str):
            pars = [pars]
        if not isinstance(pars, (list, tuple)):
            raise ValueError("Halo parameters must be specified as a list.")
        pars = list(pars)
        if not all(isinstance(p, str) for p in pars):
            raise ValueError("All halo parameters must be str.")
        self._halos_parameters = pars

    @abstractmethod
    def proxy(self, halos, theta):
        """Calculates the halo proxy for halos specified in ``halos``,
        ``theta`` is a dictionary of proxy parameters."""
        pass
