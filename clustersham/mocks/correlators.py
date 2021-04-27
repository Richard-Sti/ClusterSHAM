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

import sys
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
    verbose : bool, optional
        Jackknife verbosity flag. By default `True`.
    """
    name = "Correlator"

    def __init__(self, rpbins, pimax, boxsize, subside, Nmult=50,
                 verbose=True):
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
        self._verbose = None
        # And store these
        self.rpbins = rpbins
        self.boxsize = boxsize
        self.subside = subside
        self.pimax = pimax
        self.Nmult = Nmult
        self.verbose = verbose
        # Number of jackknifes grids
        self._Nsubs = int(self.boxsize / self.subside)

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

    @property
    def verbose(self):
        """
        Jackknife verbosity flag.

        Returns
        -------
        verbose : bool
            Verbosity.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Sets `verbose`, ensuring it is a boolean.
        """
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a bool.")
        self._verbose = verbose

    def _flush_cache(self):
        """
        Flushes the stored RR counts and resets the stored random state.
        """
        self._cache.clear()
        self._random_state = None

    def mock_wp(self, x, y, z, return_rpavg=False, nthreads=1):
        r"""
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
        return_rpavg : bool, optional
            Whether to return the mean :math:`r_p` bin separation. By default
            `False`.

        Returns
        -------
        wp : numpy.ndarray
            2-point correlation function in bins `self.rpbins`.
        rpavg : numpy.ndarray
            Average bin separation. Returned if `return_rpavg`.
        """
        result = Corrfunc.theory.wp(boxsize=self.boxsize, pimax=self.pimax,
                                    nthreads=nthreads, binfile=self.rpbins,
                                    X=x, Y=y, Z=z, output_rpavg=return_rpavg)
        if return_rpavg:
            return result['wp'], result['rpavg']
        return result['wp']

    def _count_pairs(self, x1, y1, z1, x2=None, y2=None, z2=None,
                     return_rpavg=False, nthreads=1):
        r"""
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
        return_rpavg : bool, optional
            Whether to return the mean :math:`r_p` bin separation. By default
            `False`.
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
                                      Z2=z2, periodic=False,
                                      output_rpavg=return_rpavg)

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
        edges = numpy.arange(0, self.boxsize, self.subside)
        bins = (numpy.digitize(y, edges) - 1) * self._Nsubs
        bins += numpy.digitize(x, edges) - 1
        return bins

    def nearby_bins(self, i):
        """
        Returns indices of neighbouring bins in the x-y plane.

        Parameters
        ----------
        i : int
            Central's bin index.

        Returns
        -------
        indices : numpy.ndarray
            Array of nearby bin indices. At most 8 neighbours.
        """
        # Wrap the 1D index on a 2D grid
        ix = i % self._Nsubs
        iy = i // self._Nsubs
        neighbours = numpy.array([[ix+1, iy],
                                  [ix-1, iy],
                                  [ix, iy+1],
                                  [ix, iy-1],
                                  [ix+1, iy+1],
                                  [ix+1, iy-1],
                                  [ix-1, iy-1],
                                  [ix-1, iy+1]])
        mask = numpy.logical_and(neighbours >= 0, neighbours < self._Nsubs)
        mask = numpy.logical_and(mask[:, 0], mask[:, 1])
        bins = neighbours[mask]
        return bins[:, 1] * self._Nsubs + bins[:, 0]

    def _pairs_across(self, ibin, x1, y1, z1, bins1, x2=None, y2=None, z2=None,
                      bins2=None, nthreads=1):
        """
        Calculates pair-counts between points in the central bin and in the
        neighbouring bins. Central box points will be selected from set `1`
        and neighbouring points will be selected from set `2`.

        If only set `1` is provided calculates pairs between points `1` in the
        central box and points `1` in the neighbouring bins.

        Parameters
        ----------
        x1 : numpy.ndarray
            Galaxy positions `1` along the x-axis.
        y1 : numpy.ndarray
            Galaxy positions `1` along the y-axis.
        z1 : numpy.ndarray
            Galaxy positions `1` along the z-axis.
        bins1 : numpy.ndarray
            Bin indices of set `1`.
        x2 : numpy.ndarray, optional Galaxy positions `2` along the x-axis.
        y2 : numpy.ndarray, optional
            Galaxy positions `2` along the y-axis.
        z2 : numpy.ndarray, optional
            Galaxy positions `2` along the z-axis.
        bins1 : numpy.ndarray, optional
            Bin indices of set `2`.
        nthreads : int, optional
            Number of threads. By default 1.

        Returns
        -------
        pair_counts : numpy.ndarray
            Pair counts as returned by `Corrfunc.theory.DDrppi`.
        """
        nearby_bins = self.nearby_bins(ibin)
        box_mask = bins1 == ibin

        if x2 is None or y2 is None or z2 is None:
            nearby_mask = numpy.isin(bins1, nearby_bins)
            return self._count_pairs(x1[box_mask], y1[box_mask], z1[box_mask],
                                     x1[nearby_mask], y1[nearby_mask],
                                     z1[nearby_mask], nthreads=nthreads)
        else:
            nearby_mask = numpy.isin(bins2, nearby_bins)
            return self._count_pairs(x1[box_mask], y1[box_mask], z1[box_mask],
                                     x2[nearby_mask], y2[nearby_mask],
                                     z2[nearby_mask], nthreads=nthreads)

    def mock_jackknife_cov(self, x, y, z, return_rpavg=False, nthreads=1):
        """
        Calculates the jackknife covariance error on a mock galaxy catalogue.
        In the first step, both data and random pairs are counted in the whole
        simulation box. In the second step, pairs are counted in each
        subvolume. Lastly, pairs crossing the central bin are counted. The i-th
        jackknife 2-point correlation function is obtained by subtracting the
        i-th subvolume pairs from the box pairs.

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
        return_rpavg : bool, optional
            Whether to return the mean :math:`r_p` bin separation. By default
            `False`.

        Returns
        -------
        cov : numpy.ndarray
            Jackknife covariance matrix of shape (`self.Nrpbins`,
            `self.Nrpbins`).
        wp : numpy.ndarray
            2-point correlation function of shape (`self.Nrpbins`, ).
        rpavg : numpy.ndarray
            Average bin separation. Returned if `return_rpavg`.
        """
        Nd = x.size
        Nr = Nd * self.Nmult

        try:
            if self._cache['Nd_last'] != Nd:
                self._flush_cache()
                if self.verbose:
                    print("Flushing the cache.")
                    sys.stdout.flush()
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
            if self.verbose:
                print("Counting the global RR pairs.")
                sys.stdout.flush()
            RRbox = self._count_pairs(xrand, yrand, zrand, nthreads=nthreads)
            self._cache.update({'RRbox': RRbox})

        DDbox = self._count_pairs(x, y, z, nthreads=nthreads,
                                  return_rpavg=return_rpavg)
        DRbox = self._count_pairs(x, y, z, xrand, yrand, zrand,
                                  nthreads=nthreads)

        # Used to store the jackknife 2point functions
        wps = numpy.zeros(shape=(self._Nsubs**2, self._Nrpbins))

        # Count pairs in each subvolume and substract from the box
        # statistics to obtain jackknifed 2-point correlator estimate
        bins_data = self._bin_points(x, y)
        try:
            bins_rand = self._cache['bins_rand']
        except KeyError:
            bins_rand = self._bin_points(xrand, yrand)
            self._cache.update({'bins_rand': bins_rand})

        for i in range(self._Nsubs**2):
            data_mask = bins_data == i
            rand_mask = bins_rand == i
            # RR pairs inside the subbox. Cache it!
            try:
                RRsubbox = self._cache['RRsubbox_{}'.format(i)]
            except KeyError:
                RRsubbox = self._count_pairs(
                        xrand[rand_mask], yrand[rand_mask], zrand[rand_mask],
                        nthreads=nthreads)
                self._cache.update({'RRsubbox_{}'.format(i): RRsubbox})
            # Get DD and DR pairs inside the subbox
            DDsubbox = self._count_pairs(x[data_mask], y[data_mask],
                                         z[data_mask], nthreads=nthreads)
            DRsubbox = self._count_pairs(
                    x[data_mask], y[data_mask], z[data_mask],
                    xrand[rand_mask], yrand[rand_mask], zrand[rand_mask],
                    nthreads=nthreads)
            # DD pairs crossing the subvolume
            DDacross = self._pairs_across(i, x, y, z, bins_data,
                                          nthreads=nthreads)
            # DR and RD pairs crossing the subvolume. The two differ!
            DRacross = self._pairs_across(i, x, y, z, bins_data,
                                          xrand, yrand, zrand, bins_rand,
                                          nthreads=nthreads)
            RDacross = self._pairs_across(i, xrand, yrand, zrand, bins_rand,
                                          x, y, z, bins_data,
                                          nthreads=nthreads)
            # RR pairs crossing the subvolume. Cache it!
            try:
                RRacross = self._cache['RRacross_{}'.format(i)]
            except KeyError:
                RRacross = self._pairs_across(i, xrand, yrand, zrand,
                                              bins_rand)
                self._cache.update({'RRacross_{}'.format(i): RRacross})

            # Ugly but unnecessary to wrap this in another function
            Nd_jack = Nd - data_mask.sum()
            Nr_jack = Nr - rand_mask.sum()
            DD_jack = self._subtract_counts(DDbox, DDsubbox, DDacross)
            DR_jack = self._subtract_counts(DRbox, DRsubbox, DRacross)

            RD_jack = self._subtract_counts(DRbox, DRsubbox, RDacross)

            RR_jack = self._subtract_counts(RRbox, RRsubbox, RRacross)

            wps[i, :] = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                    Nd_jack, Nd_jack, Nr_jack, Nr_jack, DD_jack, DR_jack,
                    RD_jack, RR_jack, nrpbins=self._Nrpbins, pimax=self.pimax)

        # The jackknife covariance matrix
        cov = numpy.cov(wps, rowvar=False, bias=True) * (self._Nsubs**2 - 1)
        wp = numpy.mean(wps, axis=0)
        if return_rpavg:
            rpavg = numpy.zeros(self._Nrpbins)
            for i, rmin in enumerate(numpy.unique(DDbox['rmin'])):
                mask = DDbox['rmin'] == rmin
                rpavg[i] = numpy.average(DDbox['rpavg'][mask],
                                         weights=DDbox['npairs'][mask])

            return cov, wp, rpavg
        return cov, wp

    @staticmethod
    def _subtract_counts(box_counts, subbox_counts, nearby_counts):
        """
        Subtracts the number of pairs in `subbox_counts` and `nearby_counts`
        from  `box_counts`.

        Parameters
        ----------
        box_counts : numpy.ndarray
            Pair counts from `Corrfunc.theory.DDrppi`.
        subbox_counts : numpy.ndarray
            Pair counts from `Corrfunc.theory.DDrppi`.
        nearby_counts: numpy.ndarray
            Pair counts from `Corrfunc.theory.DDrppi`.

        Returns
        -------
        result : numpy.ndarray.
            Difference of pair counts.
        """
        result = numpy.copy(box_counts)
        result['npairs'] -= (subbox_counts['npairs'] + nearby_counts['npairs'])
        return result
