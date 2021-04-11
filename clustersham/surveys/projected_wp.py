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

"""Calculates the survey projected 2-point correlation function."""

from time import perf_counter
import numpy
import Corrfunc
from kmeans_radec import kmeans_sample

from scipy.constants import c as speed_of_light


class ProjectedCorrelationFunction:
    """A class for calculating the projected two-point correlation function
    on survey data.


    Discuss reshuffling

    Comment on cosmology and units

    Parameters
    ----------
    rpbins: numpy.ndarray
        Projected orthogonal to the line of sight distance bins in which the
        projected correlation function is calculated.
    pimax: float
        Maximum distance along the line of sight to integrate over when
        calculating the projected wp.
    Njack: int
        Number of jackknifes.
    Nmult: int
        How many times more randoms used to relative to data.


    Write something that checks whether we have degrees.
    """

    def __init__(self, rpbins, pimax, Njack):
        self._rpbins = None
        self._pimax = None
        self._Njack = None
        self.rpbins = rpbins
        self.Njack = Njack
        self.pimax = pimax

    @property
    def rpbins(self):
        """Projected ortogonal to the line of sight :math`r_b`: bins."""
        return self._rpbins

    @rpbins.setter
    def rpbins(self, rpbins):
        """Sets `rpbins`."""
        if not isinstance(rpbins, numpy.ndarray) or rpbins.ndim != 1:
            raise ValueError("`rpbins` must be a 1D array of numpy.ndarray "
                             "type. Currently {}".format(type(rpbins)))
        self._rpbins = rpbins

    @property
    def Njack(self):
        """Number of jackknife clusters."""
        return self._Njack

    @Njack.setter
    def Njack(self, Njack):
        """Sets `Njack`."""
        if not isinstance(Njack, int):
            raise ValueError("`Njack` must be of int type. Currently {}"
                             .format(type(Njack)))
        self._Njack = Njack

    @property
    def pimax(self):
        """Maximum distance along the line of sight to integrate over."""
        return self._pimax

    @pimax.setter
    def pimax(self, pimax):
        """Sets `pimax`. Promotes integer values to floats."""
        if not isinstance(pimax, (int, float)):
            raise ValueError("`pimax` must be of float type. Currently {}"
                             .format(type(pimax)))
        if isinstance(pimax, int):
            pimax = float(pimax)
        self._pimax = pimax

    @staticmethod
    def _check_arrays(*Xs):
        """
        Ensures arrays Xs are of `numpy.ndarray` 1D type and have the same
        size.

        Parameters
        ----------
        X : numpy.ndarray
            Array (expected) to be checked.

        Returns
        -------
         : None
            Raises a ValueError if `Xs` contains an unknown type or do not
            have the sampe size.
        """
        if any(not isinstance(X, numpy.ndarray) or X.ndim != 1 for X in Xs):
            raise ValueError("`Xs` must be of numpy.ndarray type (1D).")

        if not all(Xs[0].size == X.size for X in Xs[1:]):
            raise ValueError("`Xs` must have the same size.")

    def _get_clusters(self, RA, DEC, Npoints):
        """
        Calculates the k-means clusters on a sphere. Number of clusters
        is `self.Njack`.

        Calls `kmeans_radec.kmeans_sample`.

        Parameters
        ----------
        RA : numpy.ndarray
            Right ascension.
        DEC : numpy.ndarray
            Declination
        Npoints : int, optional
            Number of points to predict the clusters' centres. By default
            `None`, every RA and DEC point is used.

        Returns
        -------
        kmeans_sample : `kmeans_radec.KMeans` object
            K-means object used to assign new points to the closest cluster.
        """
        mask = numpy.ones_like(RA, dtype=bool)
        if Npoints is None:
            pass
        elif Npoints > RA.size:
            raise ValueError("`Npoints` > number of data (or randoms).")
        else:
            arange = numpy.arange(RA.size)
            indx = numpy.random.choice(arange, RA.size - Npoints, replace=False)
            mask[indx] = False


        X = numpy.vstack([RA[mask], DEC[mask]]).T
        print(X.shape, self.Njack)
        return kmeans_sample(X, self.Njack, maxiter=250, tol=1e-5, verbose=0)

    @staticmethod
    def _get_nearest(kmeans, RA, DEC):
        """
        Finds the closest cluster on a 2-sphere given a fitted `kmeans`
        object.

        Parameters
        ----------
        kmeans_sample : `kmeans_radec.KMeans` object
            A fitted k-means object. See `self._get_clusters` regarding
            fitting.
        RA : numpy.ndarray
            Right ascension.
        DEC : numpy.ndarray
            Declination

        Returns
        -------
        result : numpy.ndarray
            Indices of the closest clusters.
        """
        X = numpy.vstack([RA, DEC]).T
        return kmeans.find_nearest(X)

    def _count_pairs(self, RA1, DEC1, Z1, RA2=None, DEC2=None, Z2=None,
                      cosmology=2, nthreads=1):
        r"""
        Counts 2D pair counts for the projected correlation function. If
        `RA2`, `DEC2`, and `Z2` are provided calculates cross-correlation
        between data set 1 and data set 2.

        Calls `Corrfunc.mocks.DDrppi_mocks`.


            .. Note:
                All particles are assumed to have a uniform weight.


        Parameters
        ----------
        RA1 : numpy.ndarrray
            Right ascension in degrees (:math:`0 < RA < 360`).
        DEC1 : numpy.ndarrray
            Declination in degrees (:math:`-90 < DEC < 90`).
        Z1 : numpy.ndarray
            Redshift.
        RA2 : numpy.ndarrray, optional.
            Right ascension in degrees (:math:`0 < RA < 360`). By default
            `None`, i.e. autocorrelation regime.
        DEC2 : numpy.ndarrray
            Declination in degrees (:math:`-90 < DEC < 90`). By default `None`,
            i.e. autocorrelation regime.
        Z2 : numpy.ndarray
            Redshift. By default `None`, i.e. autocorrelation regime.
        cosmology : int, optional
            Cosmology integer choice. Valid values are `1` (LasDamas cosmology
            :math:`\Omega_m = 0.25`, :math:`\Omega_\Lambda = 0.75`) and `2`
            (Planck cosmology :math:`\Omega_m=0.302`,
            :math:`\Omega_\Lambda = 0.698`). For how to add different
            cosmologies see Corrfunc documentation.
        nthreads : int, optional
            Number of openMP threads, if allowed. By default 1.

        Returns
        -------
        counts : structured numpy.ndarray
            Structured array with 2D pair counts.
        """
        if RA2 is None or DEC2 is None or Z2 is None:
            weights1 = numpy.ones_like(RA1)
            CZ1 = speed_of_light * 1e-3 * Z1 # comment here
            return Corrfunc.mocks.DDrppi_mocks(
                    autocorr=1, cosmology=cosmology, nthreads=nthreads,
                    pimax=self.pimax, binfile=self.rpbins, RA1=RA1, DEC1=DEC1,
                    CZ1=CZ1, weights1=weights1, weight_type='pair_product',
                    output_rpavg=True)
        else:
            weights1 = numpy.ones_like(RA1)
            weights2 = numpy.ones_like(RA2)
            CZ1 = speed_of_light * 1e-3 * Z1
            CZ2 = speed_of_light * 1e-3 * Z2
            return Corrfunc.mocks.DDrppi_mocks(
                    autocorr=0, cosmology=cosmology, nthreads=nthreads,
                    pimax=self.pimax, binfile=self.rpbins, RA1=RA1, DEC1=DEC1,
                    CZ1=CZ1, RA2=RA2, DEC2=DEC2, CZ2=CZ2, weights1=weights1,
                    weights2=weights2, weight_type='pair_product')

    def _count_DDDRRR(self, RA, DEC, Z, randRA, randDEC, randZ, cosmology, nthreads):
        # DOCS
        DD = self._count_pairs(RA, DEC, Z, cosmology=cosmology,
                               nthreads=nthreads)
        DR = self._count_pairs(RA, DEC, Z, randRA, randDEC, randZ,
                               cosmology=cosmology, nthreads=nthreads)
        RR = self._count_pairs(randRA, randDEC, randZ, cosmology=cosmology,
                               nthreads=nthreads)
        return DD, DR, RR


    @staticmethod
    def random_redshift(Z, N):
        return numpy.random.choice(Z, N, replace=True)

    def survey_wp(self, RA, DEC, Z, randRA, randDEC, cosmology=2,
                  nthreads=1, seed=42):
        """

        ADd seeds docum

        Calculates the survey correlation function using the Landy-Szalay
        estimater.

        Calls `Corrfunc.utils.convert_rp_pi_counts_to_wp` to convert pair
        counts from `Corrfunc.mocks.DDrppi_mocks`.


            .. Note:
                All particles are assumed to have a uniform weight.


        Output rpavg as well? Start doing this by default

        Parameters
        ----------
        RA : numpy.ndarrray
            Right ascension in degrees (:math:`0 < RA < 360`).
        DEC : numpy.ndarrray
            Declination in degrees (:math:`-90 < DEC < 90`).
        Z : numpy.ndarray
            Redshift.
        randRA : numpy.ndarrray, optional.
            Randoms right ascension in degrees (:math:`0 < RA < 360`).
        randDEC2 : numpy.ndarrray
            Randoms declination in degrees (:math:`-90 < DEC < 90`).
        cosmology : int, optional
            Cosmology integer choice. Valid values are `1` (LasDamas cosmology
            :math:`\Omega_m = 0.25`, :math:`\Omega_\Lambda = 0.75`) and `2`
            (Planck cosmology :math:`\Omega_m=0.302`,
            :math:`\Omega_\Lambda = 0.698`). For how to add different
            cosmologies see Corrfunc documentation.
        nthreads : int, optional
            Number of openMP threads, if allowed. By default 1.

        Returns
        -------
        wp : numpy.ndarray
            Survey correlation function in bins `self.rpbbins`.
        """
        numpy.random.seed(seed)
        # Check input arrays
        self._check_arrays(RA, DEC, Z)
        self._check_arrays(randRA, randDEC)
        # Get the random redshifts using the reshuffle method.
        randZ = self.random_redshift(Z, randRA.size)
        # Pair counting
        DD, DR, RR = self._count_DDDRRR(RA, DEC, Z, randRA, randDEC, randZ,
                                        cosmology, nthreads)
        # Convert pair counts to the correlation function
        Nd = RA.size
        Nr = randRA.size
        wp = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                ND1=Nd, ND2=Nd, NR1=Nr, NR2=Nr, D1D2=DD, D1R2=DR, D2R1=DR,
                R1R2=RR, pimax=self.pimax, nrpbins=self.rpbins.size - 1)

        return {'rpavg': self._get_rpavg(DD),
                'wp': wp,
                'pimax': self.pimax,
                'rpbins': self.rpbins}

    def _get_rpavg(self, DD):
        rpavg = numpy.zeros(self.rpbins.size - 1)
        rmins = numpy.unique(DD['rmin'])
        for i, rmin in enumerate(rmins):
            mask = DD['rmin'] == rmin
            rpavg[i] = numpy.average(DD['rpavg'][mask], weights=DD['npairs'][mask])
        return rpavg

    def survey_jackknife(self, RA, DEC, Z, randRA, randDEC, cosmology=2,
                         nthreads=1, verbose=1, Npoints_kmeans=None, seed=42):
        """Explain why cannot just subtract pairs"""
        numpy.random.seed(seed)
        # Check input arrays
        self._check_arrays(RA, DEC, Z)
        self._check_arrays(randRA, randDEC)

        if Npoints_kmeans is None:
            Npoints_kmeans = RA.size

        # Calculate kmeans first on randoms change the number of points to choice
        kmeans = self._get_clusters(randRA, randDEC, Npoints=Npoints_kmeans)

        data_bins = self._get_nearest(kmeans, RA, DEC)
        rand_bins = self._get_nearest(kmeans, randRA, randDEC)
        if verbose:
            print('Calculated k-means clusters.')

        randZ = self.random_redshift(Z, randRA.size)


        if verbose:
            timer = numpy.full(self.Njack, fill_value=numpy.nan)

        wps = numpy.zeros(shape=(self.Njack, self.rpbins.size - 1))
        for i in range(self.Njack):
            start = perf_counter()
            mask_data = data_bins != i
            mask_rand = rand_bins != i
            DD, DR, RR = self._count_DDDRRR(
                    RA[mask_data], DEC[mask_data], Z[mask_data],
                    randRA[mask_rand], randDEC[mask_rand], randZ[mask_rand],
                    cosmology, nthreads)

            Nd = mask_data.sum()
            Nr = mask_rand.sum()

            wps[i, :] = Corrfunc.utils.convert_rp_pi_counts_to_wp(
                    ND1=Nd, ND2=Nd, NR1=Nr, NR2=Nr, D1D2=DD, D1R2=DR, D2R1=DR,
                    R1R2=RR, pimax=self.pimax, nrpbins=self.rpbins.size - 1)

            # Time keeper, estimates remaining time
            if verbose:
                timer[i] = perf_counter() - start
                remaining_time = numpy.mean(timer[~numpy.isnan(timer)])
                remaining_time *= (self.Njack - i + 1)
                if remaining_time < 60:
                    unit = 'seconds'
                elif remaining_time < 3600:
                    remaining_time /= 60
                    unit = 'minutes'
                else:
                    remaining_time /= 3600
                    unit = 'hours'

                print("Completed {}/{}. Estimated remaining time {:.2f} {}"
                      .format(i+1, self.Njack+1, remaining_time, unit))

        jack_wp = numpy.cov(wps, rowvar=False, bias=True) * (self.Njack - 1)
        # Want to know rpavg, need DD for the whole volume
        DD = self._count_pairs(RA, DEC, Z, cosmology=cosmology,
                               nthreads=nthreads)
        return {'rpavg': self._get_rpavg(DD),
                'wp': numpy.mean(wps, axis=0),
                'cov': jack_wp,
                'pimax': self.pimax,
                'rpbins': self.rpbins}
