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

"""
Correlator to calculate the 2-point correlation function and jackknife
covariance matrix.
"""

import numpy
import Corrfunc


class Correlator:
    """
    A class providing a simple interface to calculate the 2-point projected
    correlation function and the jackknife covariance matrix.

    When calculating the jackknife covariance matrix caches the RR pair counts
    and upon every call to `self.mock_jackknife_cov` generates identical
    randoms. This is relatively inexpensive compared to the the price for
    storing the randoms consistently in the memory, which may be required
    elsewhere at runtime.

    .. Note:
        The simulation box must be cornered at the origin.


    Parameters
    ----------
        bins specified in ``rpbins``.
    rpbins : numpy.ndarray
        Array of bins projected orthogonal to the line of sight in which
        to calculate the 2-point projected correlation function.
    pimax : float
        Maximum distance along the line of sight to integrate over when
        calculating the 2-point projected correlation function.
    boxsize : int
        The side length of the simulation box.
    subside : int
        The subvolume side length in the x-y plane used for jackknifing.
    Nmult : int, optional
        How many more randoms than data in the jackknife calculation. By
        default 50.
    """

    def __init__(self, rpbins, pimax, boxsize, subside, Nmult=50):
        # Caching RR pair counts and random state
        self._cache = {}
        self._random_state = None

        # Initialise a bunch of place holders
        self._rpbins = None
        self._Nrpbins = None
        self._pimax = None
        self._subside = None
        self._Nmult = None
        self._boxsize = None
        # And store these
        self.rpbins = rpbins
        self.boxsize = boxsize
        self.subside = subside
        self.pimax = pimax
        self.Nmult = Nmult
        # Number of jackknifes
        self._Nsubs= int(self.boxsize / self.subside)**2

    @property
    def rpbins(self):
        """
        Bins in the X-Y plane to calculate the 2-point projected correlation
        function.
        """
        return self._rpbins

    @rpbins.setter
    def rpbins(self, rpbins):
        """Sets `rpbins`."""
        if rpbins.ndim != 1:
            raise ValueError("`rpbins` must be of 1D numpy.ndarray type.")
        self._flush_cache()
        self._rpbins = rpbins
        self._Nrpbins = rpbins.size - 1


    @property
    def Nrpbins(self):
        """The number of radially projected bins."""
        return self._Nrpbins

    @property
    def pimax(self):
        """The maximum integration distance along the Z-axis."""
        return self._pimax

    @pimax.setter
    def pimax(self, pimax):
        """Sets `pimax`."""
        if not isinstance(pimax, (int, float)):
            raise ValueError("`pimax` must be either an int or a float.")
        self._flush_cache()
        self._pimax = pimax

    @property
    def boxsize(self):
        """Simulation box side length."""
        return self._boxsize

    @boxsize.setter
    def boxsize(self, L):
        """
        Sets `boxsize`, ensures it is an integer for jackknifing. This
        requirement could be lifted in the future if necessary.
        """
        if not isinstance(L, int):
            raise ValueError("'boxsize' must be of int type")
        self._boxsize = L

    @property
    def subside(self):
        r"""
        Side :math:`d` of the subvolume removed at each turn. If the box
        side is :math:`L`, then the removed subvolume is :math:`d^2 * L`.
        """
        return self._subside

    @subside.setter
    def subside(self, subside):
        if not isinstance(subside, int):
            raise ValueError('`subside` must be an integer.')
        if not self.boxsize % subside == 0:
            raise ValueError('`subside` must divide `self.boxsize`.')
        self._flush_cache()
        self._subside = subside

    @property
    def Nmult(self):
        """
        How many times more randoms than data to generate when calculating
        jackknifes.
        """
        return self._Nmult

    @Nmult.setter
    def Nmult(self, N):
        """Sets `Nmult`."""
        if not isinstance(N, int):
            raise ValueError("'Nmult' must be of int type.")
        self._Nmult = N

    def _flush_cache(self):
        """
        Flushes the stored RR counts and resets the stored random state.
        """
        self._cache.clear()
        self._random_state = None

    def mock_wp(self, x, y, z, nthreads=1):
        """
        Calculates the 2-point correlation function in a simulation box.
        Calls `Corrfunc.theory.wp`. The output is in whatever units the
        galaxy positions are given.

        Parameters
        ----------
        x : numpy.ndarray
            Galaxy positions along the x-axis.
        y : numpy.ndarray
            Galaxy positions along the y-axis.
        z : numpy.ndarray
            Galaxy positions along the z-axis.
        nthreads : int, optional
            Number of threads. By default 1.

        Returns
        -------
        result : numpy.ndarray
            2-point correlation function in bins `self.rpbins`.
        """
        return Corrfunc.theory.wp(boxsize=self.boxsize, pimax=self.pimax,
                                  nthreads=nthreads, binfile=self.rpbins,
                                  X=x, Y=y, Z=z)['wp']

    def _count_pairs(self, x1, y1, z1, x2=None, y2=None, z2=None, nthreads=1):
        """
        Counts 3D galaxy pairs in a simulation box. If `x2`, or `y2`, or `z2`
        are given turns performs cross-correlation between 1 and 2.
        Calls `Corrfunc.theory.DDrppi`.

        Parameters
        ----------
        x1 : numpy.ndarray
            Galaxy positions along the x-axis.
        y1 : numpy.ndarray
            Galaxy positions along the y-axis.
        z1 : numpy.ndarray
            Galaxy positions along the z-axis.
        x2 : numpy.ndarray
            Galaxy (or random) positions along the x-axis.
        y2 : numpy.ndarray
            Galaxy (or random) positions along the y-axis.
        z2 : numpy.ndarray
            Galaxy (or random) positions along the z-axis.
        nthreads : int, optional
            Number of threads. By default 1.

        Returns
        -------
        pair_counts : numpy.ndarray
            Pair counts as returned by `Corrfunc.theory.DDrppi`.
        """
        if x2 is None or y2 is None or z2 is None:
            autocorr = True
        else:
            autocorr = False
        return Corrfunc.theory.DDrppi(autocorr, nthreads=nthreads,
                                      pimax=self.pimax, binfile=self.rpbins,
                                      X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2,
                                      Z2=z2, periodic=False)

    def _get_randoms(self, N):
        """
        Samples `N` uniformly distributed points inside the simulation box.
        Temporarily switches the random number generator back to its initial
        state to ensure that the newly sampled points match the ones that
        were previously used to calculate cached RRs.

        Parameters
        ----------
        N : int
            Number of points to generate.

        Returns
        -------
        randoms : numpy.ndarray
            Random points. Shape is (3, `N`)
        """
        # Either cache the random generator state or temporarily set it back
        if self._random_state is None:
            self._random_state = numpy.random.get_state()
        else:
            current_state = numpy.random.get_state()
            numpy.random.set_state(self._random_state)
        # Sample the randoms
        randoms = numpy.random.uniform(0, self.boxsize, size=(3, N))
        # Switch back the random generator, if it was set back previously
        try:
            numpy.random.set_state(current_state)
        except UnboundLocalError:
            pass

        return randoms

    def _bin_points(self, x, y):
        """
        Bins points in the x- and y-plane in bins of size `self.subside`
        squared.

        Parameters
        ----------
        x : numpy.ndarray
            Points' coordinates along the x-axis.
        y : numpy.ndarray
            Points' coordinates along the y-axis.

        Returns
        -------
        bins : numpy.ndarray
            Each point's bin.
        """
        edges = numpy.arange(0, self.boxsize + self.subside, self.subside)
        bins = (numpy.digitize(y, edges) - 1) * (edges.size - 1)
        bins += numpy.digitize(x, edges) - 1
        return bins

    def mock_jackknife_cov(self, x, y, z, return_wp=False, nthreads=1):
        """
        Calculates the jackknife covariance error on a mock galaxy catalogue.
        In the first step, both data and random pairs are counted in the whole
        simulation box. In the second step, pairs are counted in each
        subvolume. The i-th jackknife 2-point correlation function is obtained
        by substracting the i-th subvolume pairs from the box pairs.

        Caches RR pair results when first called.

        Parameters
        ----------
        x : numpy.ndarray
            Galaxy positions along the x-axis.
        y : numpy.ndarray
            Galaxy positions along the y-axis.
        z : numpy.ndarray
            Galaxy positions along the z-axis.
        nthreads : int, optional
            Number of threads. By default 1.
        return_wp : bool, optional
            Whether to return the mean 2-point correlation function.
            By default False, not returned.

        Returns
        -------
        cov : numpy.ndarray
            Jackknife covariance matrix of shape (`self.Nrpbins`,
            `self.Nrpbins`).
        wp : numpy.ndarray
            2-point correlation function of shape (`self.Nrpbins`, ).
            Returned if `return_wp`.
        """
        Nd = x.size
        Nr = Nd * self.Nmult

        try:
            if self._cache['Nd_last'] != Nd:
                self._flush_cache()
        except KeyError:
            if numpy.min(x) < 0 or numpy.min(y) < 0 or numpy.min(z) < 0:
                raise ValueError("The simulation box must be cornered "
                                 "at the origin.")
        # Takes extra care of the random number generator
        xrand, yrand, zrand = self._get_randoms(Nr)

        # First count pairs in the whole box
        try:
            RRbox = self._cache['RRbox']
        except KeyError:
            RRbox = self._count_pairs(xrand, yrand, zrand, nthreads=nthreads)
            self._cache.update({'RRbox': RRbox})

        DDbox = self._count_pairs(x, y, z, nthreads=nthreads)
        DRbox = self._count_pairs(x, y, z, xrand, yrand, zrand,
                                  nthreads=nthreads)

        # Used to store the jackknife 2point functions
        wps = numpy.zeros(shape=(self._Nsubs, self._Nrpbins))

        # Count pairs in each subvolume and substract from the box
        # statistics to obtain jackknifed 2-point correlator estimate
        bins_data = self._bin_points(x, y)
        try:
            bins_rand = self._cache['bins_rand']
        except KeyError:
            bins_rand = self._bin_points(xrand, yrand)
            self._cache.update({'bins_rand': bins_rand})

        for i in range(self._Nsubs):
            data_mask = bins_data == i
            rand_mask = bins_rand == i

            try:
                RRsubbox = self._cache['RRsubbox_{}'.format(i)]
            except KeyError:
                RRsubbox = self._count_pairs(
                        xrand[rand_mask], yrand[rand_mask], zrand[rand_mask],
                        nthreads=nthreads)
                self._cache.update({'RRsubbox_{}'.format(i): RRsubbox})

            DDsubbox = self._count_pairs(x[data_mask], y[data_mask],
                                         z[data_mask], nthreads=nthreads)
            DRsubbox = self._count_pairs(
                    x[data_mask], y[data_mask], z[data_mask],
                    xrand[rand_mask], yrand[rand_mask], zrand[rand_mask],
                    nthreads=nthreads)
            # Ugly but unnecessary to wrap this in another function
            Nd_jack = Nd - data_mask.sum()
            Nr_jack = Nr - rand_mask.sum()
            DD_jack = self._substract_counts(DDbox, DDsubbox)
            DR_jack = self._substract_counts(DRbox, DRsubbox)
            RR_jack = self._substract_counts(RRbox, RRsubbox)

            wps[i, :] = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                    Nd_jack, Nd_jack, Nr_jack, Nr_jack, DD_jack, DR_jack,
                    DR_jack, RR_jack, nrpbins=self._Nrpbins, pimax=self.pimax)

        # The jackknife covariance matrix
        cov = numpy.cov(wps, rowvar=False, bias=True) * (self._Nsubs - 1)
        if return_wp:
            wp = numpy.mean(wps, axis=0)
            return cov, wp
        return cov

    @staticmethod
    def _substract_counts(box_counts, subbox_counts):
        """
        Substracts the number of pairs in `subbox_counts` from  `box_counts`.

        Parameters
        ----------
        box_counts : numpy.ndarray
            Pair counts from `Corrfunc.theory.DDrppi`.
        subbox_counts : numpy.ndarray
            Pair counts from `Corrfunc.theory.DDrppi`.

        Returns
        -------
        result : numpy.ndarray.
            Difference of pair counts.
        """
        result = numpy.copy(box_counts)
        result['npairs'] -= subbox_counts['npairs']
        return result
