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

"""
Likelihood classes. Currently only a Gaussian likelihood on the 2-point
correlation function, comparing clustering in mock galaxy catalogs to
observation.

In the future may include a conditional stellar mass function and others.
"""

from abc import (ABC, abstractmethod)
import numpy
from scipy.stats import multivariate_normal


class BaseLikelihood(ABC):
    """
    A base likelihood class.

        .. Note:
            Currently implementing a base class is redundant, however it may
            be useful in the future.
    """
    name = None

    @abstractmethod
    def __call__(self, **kwargs):
        """
        The log-likelihood.

        Parameters
        ----------
        **kwargs :
            Arguments passed the likelihood function.
        """
        pass



class GaussianClusteringLikelihood(BaseLikelihood):
    """
    The multivariate Gaussian likelihood. It is assumed that the mean is
    the survey correlation function and covariances are added.

    Parameters
    ----------
    wp_survey : numpy.ndarray
        Survey 2-point projected correlation function.
    cov_survey : numpy.ndarray
        Survey 2-point projected correlation function covariance matrix.
    """
    name = "GaussianClusteringLikelihood"

    def __init__(self, wp_survey, cov_survey):
        self._wp_survey = None
        self._cov_survey = None
        self.wp_survey = wp_survey
        self.cov_survey = cov_survey

    @property
    def wp_survey(self):
        """
        The survey 2-point correlation function.
        """
        return self._wp_survey

    @wp_survey.setter
    def wp_survey(self, wp):
        """Sets `wp_survey`."""
        if wp.ndim != 1:
            raise ValueError("'wp_survey' must be a numpy.ndarray (1D).")
        self._wp_survey = wp

    @property
    def cov_survey(self):
        """The covariance matrix for `self.wp_survey`."""
        return self._cov_survey

    @cov_survey.setter
    def cov_survey(self, cov):
        """Sets `cov_survey`."""
        if cov.shape != (self.wp_survey.size, self.wp_survey.size):
            raise ValueError("`wp_cov` must be of 2D numpy.ndarray type "
                             "corresponding to `self.wp_survey`.")
        self._cov_survey = cov

    def __call__(self, mock_wp, mock_stoch_cov=None, mock_jack_cov=None):
        r"""
        The multivariate Gaussian log-likelihood

            .. math::

                \log \mathcal{N}(x | \mu, \Sigma)

        where :math:`x` is `mock_wp`, :math:`mu` is `self._wp_survey`, and
        :math:`Sigma` is the sum of the survey and mock covariance matrices.

        Parameters
        ----------
        mock_wp : numpy.ndarray
            The mock 2-point correlation function.
        mock_stoch_cov : numpy.ndarray
            The mock 2-point jackknife correlation function covariance matrix.
        mock_stoch_cov : numpy.ndarray
            The mock 2-point stochastic correlation function covariance matrix.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `mock_wp`.
        """
        # Add up the covariance matrices
        cov = numpy.copy(self.cov_survey)
        if mock_stoch_cov is not None:
            cov += mock_stoch_cov
        if mock_jack_cov is not None:
            cov += mock_jack_cov
        return multivariate_normal.logpdf(mock_wp, mean=self.wp_survey,
                                          cov=cov)
