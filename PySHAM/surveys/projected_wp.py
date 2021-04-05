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
"""Projected correlation function calculation for surveys."""
from sys import stdout
from time import time

import numpy as np

from kmeans_radec import kmeans_sample

from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

from .base import BaseProjectedCorrelationFunction


class ProjectedCorrelationFunction(BaseProjectedCorrelationFunction):
    r"""A class for calculating the projected two-point correlation function
    on survey data.

    Parameters
    ----------
    data: numpy.ndarray
        Data object corresponding to the survey bin. Should be returned
        by ``PySHAM.surveys.base.BaseSurvey.scope_selection``.
    randoms: numpy.ndarray
        A precomputed numpy structured array of randoms matching the survey
        angular geometries. Entires must be ``RA`` and ``DEC``.
    rpbins: numpy.ndarray
        A projected along the line of sight distance bins in which the
        projected correlation function is calculated.
    pimax: float
        Maximum distance along the line of sight to integrate over when
        calculating the projected wp.
    Njack: int
        Number of jackknifes.
    Nmult: int
        How many times more randoms used to relative to data.
    nthreads : int, optional
        Number of threads.
    """

    def __init__(self, data, randoms, rpbins, pimax, Njack, Nmult, nthreads):
        self.Njack = Njack
        self.Nmult = Nmult
        self.data = data
        self.randoms = randoms
        self.rpbins = rpbins
        self.pimax = pimax
        self.nthreads = nthreads
        # setup the jackknifing regions
        self._setup_clusters()

    def _setup_clusters(self):
        """Splits up the survey into clusters using a k-means algorithm.
        By default assigns uniform weights to all points.
        """
        Ngals = self.data.size
        Nrands = self.randoms.size
        print('NGALS', Ngals)
        print('Nrands', Nrands)

        Xrands = np.vstack([self.randoms['RA'], self.randoms['DEC']]).T
        Xgals = np.vstack([self.data['RA'], self.data['DEC']]).T

        mask = np.random.choice(np.arange(Nrands), 3*Ngals, replace=False)
        kmeans = kmeans_sample(Xrands[mask, :], self.Njack, maxiter=250,
                               tol=1.0e-5, verbose=0)
        self.data['labels'] = kmeans.find_nearest(Xgals)
        self.randoms['labels'] = kmeans.find_nearest(Xrands)

        # make weights simply just 1
        self.data['weights'] = np.ones(Ngals)
        self.randoms['weights'] = np.ones(Nrands)

    def _leave_one_out_wp(self, index):
        """Calculates the projected correlation function after excluding the
        ``index`` cluster.
        """
        data = self.data[self.data['labels'] != index]
        rand = self.randoms[self.randoms['labels'] != index]

        Ndata = data.size
        Nrand = rand.size

        # cosmology = 2 means Planck cosmology
        DD = DDrppi_mocks(autocorr=True, cosmology=2, nthreads=self.nthreads,
                          pimax=self.pimax, binfile=self.rpbins,
                          RA1=data['RA'], DEC1=data['DEC'], CZ1=data['CZ'],
                          weights1=data['weights'], weight_type='pair_product')

        DR = DDrppi_mocks(autocorr=False, cosmology=2, nthreads=self.nthreads,
                          pimax=self.pimax, binfile=self.rpbins,
                          RA1=data['RA'], DEC1=data['DEC'], CZ1=data['CZ'],
                          RA2=rand['RA'], DEC2=rand['DEC'], CZ2=rand['CZ'],
                          weights1=data['weights'], weights2=rand['weights'],
                          weight_type='pair_product')

        RR = DDrppi_mocks(autocorr=True, cosmology=2, nthreads=self.nthreads,
                          pimax=self.pimax, binfile=self.rpbins,
                          RA1=rand['RA'], DEC1=rand['DEC'], CZ1=rand['CZ'],
                          weights1=rand['weights'], weight_type='pair_product')

        return convert_rp_pi_counts_to_wp(ND1=Ndata, ND2=Ndata, NR1=Nrand,
                                          NR2=Nrand, D1D2=DD, D1R2=DR, D2R1=DR,
                                          R1R2=RR, pimax=self.pimax,
                                          nrpbins=len(self.rpbins) - 1)

    def wp_jackknife(self):
        """Returns the mean jackknife correlation function for the survey
        and the corresponding covariance matrix.
        """
        wps = [None] * self.Njack
        execution_time = np.full(self.Njack, fill_value=np.nan)
        for i in range(self.Njack):
            start = time()
            wps[i] = self._leave_one_out_wp(i)

            execution_time[i] = time() - start
            mean_time = np.mean(execution_time[np.isfinite(execution_time)])
            remaining_time = mean_time * (self.Njack - i - 1) / 60**2
            print("Done with {}/{}. Estimated remaining time is {:.2f} hours"
                  .format(1 + i, self.Njack, remaining_time))
            stdout.flush()

        wps = np.array(wps)
        wp = np.mean(wps, axis=0)
        # bias=True means normalisation by 1/N
        cov_jack = np.cov(wps, rowvar=False, bias=True) * (self.Njack - 1)
        return wp, cov_jack
