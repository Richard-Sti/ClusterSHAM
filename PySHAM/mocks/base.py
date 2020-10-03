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
        if not rpbins.ndim != 1:
            raise ValueError("``rpbins`` must be a 1-dimensional array.")
        self._rpbins = rpbins

    @property
    def pimax(self):
        """Returns pimax, the maximum integration distance along the line of
        sight."""
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
class BaseAbundanceMatch(Base):

    r"""Abstract base class that handles inputs specific to abundanc matching.

    Parameters
    ----------
    scope
    nd_gal
    is_luminosity
    Nmocks
    halo_proxy
    halos
    xrange
    """

    _scope = None
    _nd_gal = None
    _is_luminosity = None
    _Nmocks = None
    _faint_end_first = None
    _halo_proxy = None
    _halos = None
    _xrange = None

    @property
    def scope(self):
        """Magnitude or log mass range over which to perform abundance
        matching.
        """
        return self._scope

    @scope.setter
    def scope(self, scope):
        """Sets the scope."""
        if not isinstance(scope, (list, tuple)):
            raise ValueError("Must provide a list or tuple for ``scope``.")
        scope = tuple(scope)
        if len(scope) != 2:
            raise ValueError("``scope`` must have length 2.")
        if not scope[1] > scope[0]:
            scope = scope[::-1]
        self._scope = scope
        if scope[0] > 0:
            self._is_luminosity = False
            self._faint_end_first = True
        else:
            self._is_luminosity = True
            self._faint_end_first = False

    @property
    def is_luminosity(self):
        """Returns True if luminosity abundance matching."""
        return self._is_luminosity

    @property
    def nd_gal(self):
        """Luminosity or mass function used."""
        return self._nd_gal

    @nd_gal.setter
    def nd_gal(self, nd):
        """Sets either the luminosity or mass function."""
        if not isinstance(nd, np.ndarray):
            raise ValueError("``nd`` must be of numpy.ndarray type.")
        if np.any(nd[:, 2] < 0):
            raise ValueError("``nd`` cannot be in log-densities.")
        self._nd_gal = nd

    @property
    def Nmocks(self):
        """Returns the number of mocks to produce at a given point."""
        return self._Nmocks

    @Nmocks.setter
    def Nmocks(self, Nmocks):
        """Sets the number of mocks."""
        if Nmocks < 1:
            raise ValueError("``Nmocks`` must be larger than 1.")
        self._Nmocks = int(Nmocks)

    @property
    def halo_proxy(self):
        """The specified halo proxy function. As inputs must take halos
        and **kwargs specifiying the proxy parameters."""
        return self._halo_proxy

    @halo_proxy.setter
    def halo_proxy(self, halo_proxy):
        """Sets the halo proxy."""
        self._halo_proxy = halo_proxy

    @property
    def halos(self):
        """Returns the halos object."""
        return self._halos

    @halos.setter
    def halos(self, halos):
        """Sets the halos and check whether all pars present."""
        if not isinstance(halos, np.ndarray):
            raise ValueError("``halos`` must be of numpy.ndarray type.")
        pos = ['x', 'y', 'y']
        if not all(p in halos.dtype.names for p in pos):
            raise ValueError("Halo positions must be specified as "
                             "`x`, `y`, `z`.")
        if not all(p in halos.dtype.names for p in self.halo_proxy.parameters):
            raise ValueError("``halos`` mising some halo proxy parameters.")

        # store only the relevant parameters
        pars = pos + list(self.halo_proxy.parameters)
        formats = ['float64'] * len(pars)
        N = halos['x'].size
        self._halos = np.zeros(N, dtype={'names': pars, 'formats': formats})
        for p in pars:
            self._halos[p] = halos[p]


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
    halos_parameters

    """

    _halos_parameters = None

    @property
    def halos_parameters(self):
        """Returns the halo parameters needed for the proxy calculation."""
        return self._halos_parameters

    @halos_parameters.setters
    def halos_parameters(self, pars):
        """Sets the halo parameters"""
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
