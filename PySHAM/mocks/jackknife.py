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
"""A module for *fast* estimate of the jackknife covariance matrix on a
simulation box."""

import numpy as np
import Corrfunc

from .base import Base


class Jackknife(Base):
    r""" Class for fast estimates of the jackknife covariance matrix on a
    simulation box. Based on the Corrfunc package.

    Note
    ----------
    Describe in words what the code does.


    Parameters
    ----------
    subside : int
        Lenght of a subvolume being removed at each turned. This subvolume
        is assumed to be ``subside`` x ``subside`` x ``boxsize.
    boxsize : int
        Length of a side of the simulation box in which halos are located.
    rpbins : numpy.ndarray
        Array of bins orthogonal to the line of sight in which the
        project correlation function is calculated.
    pimax : float
        Maximum distance along the line of sight to integrate over when
        calculating the projected wp.
    nthreads : int, optional
        Number of threads.
    """

    def __init__(self, subside, boxsize, rpbins, pimax, nthreads=1):
        self.boxsize = boxsize
        self.rpbins = rpbins
        self.pimax = pimax
        self.subside = subside
        self.nthreads = nthreads
        # number of bins
        self._nrpbins = len(self.rpbins) - 1

    @property
    def subside(self):
        """Length of the subvolume being removed at each turned. Assumed
        to be ``subside`` x ``subside`` x ``boxsize."""
        return self._subside

    @subside.setter
    def subside(self, subside):
        if not isinstance(subside, (float, int)):
            raise ValueError('``subside`` must be of type int.')
        if not self.boxsize % subside == 0:
            raise ValueError('``subside`` must divide ``boxsize``.')
        self._subside = int(subside)

    def _count_pairs(self, autocorr, x1, y1, z1, x2=None, y2=None, z2=None):
        """A wrapper around Corrfunc.theory.DDrppi"""
        return Corrfunc.theory.DDrppi(autocorr, nthreads=self.nthreads,
                                      pimax=self.pimax, binfile=self.rpbins,
                                      X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2,
                                      Z2=z2, periodic=False)

    def _counts2wp(self, DD, DR, RR, Nd, Nr):
        """A wrapper around Corrfunc.utils.convert_rp_pi_counts_to_wp"""
        return Corrfunc.utils.convert_rp_pi_counts_to_wp(Nd, Nd, Nr, Nr, DD,
                                                         DR, DR, RR,
                                                         nrpbins=self._nrpbins,
                                                         pimax=self.pimax)

    def _bins(self, X, Y):
        """Bins the galaxies along the z-axis into bins of size
        subside x subside. Assumes box dimensions are [0, boxsize]^3
        """
        edges = np.arange(0, self.boxsize + self.subside, self.subside)
        nboxes = edges.size - 1
        return (np.digitize(Y, edges) - 1)*nboxes + np.digitize(X, edges) - 1

    @staticmethod
    def _subtract_pairs(counts, counts_sub, weight=1.0):
        """A help fuction to substract pairs counted by Corrfunc, defined so
        as to follow the corrfunc data structure.

        Parameters:
            counts:     Corrfunc counts object
            counts_sub: Corrfunc counts object
                pairs to be subtracted
            weight:     float (optional)
                in some cases the subtracted pairs are down-weighted.
        """
        for i in range(counts.size):
            counts[i]['npairs'] -= counts_sub[i]['npairs'] * weight
            counts[i]['npairs'] = int(round(counts[i]['npairs'], 0))
        return counts

    def _randoms(self, nrand):
        """Generates random uniformly distributed over the simulation box"""
        return [np.random.uniform(0, self.boxsize, nrand) for __ in range(3)]

    def _average_subvol_rr_pairs(self, bins, x, y, z):
        """Calculates average number of RR pairs, with both points within
        the subvolume. As this is pure RR of a uniform distribution it is
        enough to calculate it once and then use this value when calculating
        each jackknife
        """
        nsubs = np.unique(bins).size
        rrsubs, npoints = list(), list()
        naverage = 10
        # Calculate for 10 random subvolumes the number of points in it and
        # count the pairs
        for i in np.random.choice(range(nsubs), size=naverage, replace=False):
            mask = np.where(bins == i)
            npoints.append(mask[0].size)
            rrsubs.append(self._count_pairs(True, x[mask], y[mask], z[mask]))
        # Average number of points within a subvolume, must be an integer
        npoints_average = int(round(sum(npoints) / len(npoints), 0))
        # Calculate the average number of RR pairs both along line of sight
        # and orthogonal to it
        rrsub_average = rrsubs[0].copy()
        # Start counting from the second subvol, first included already.
        for i in range(rrsub_average.size):
            for j in range(1, len(rrsubs)):
                rrsub_average[i]['npairs'] += (rrsubs[j])[i]['npairs']
            # Get the average
            rrsub_average[i]['npairs'] = int(round(
                rrsub_average[i]['npairs'] / len(rrsubs), 0))
        return rrsub_average, npoints_average

    def _centers(self, x, y, bins):
        """Calculates the position of the center of each bin"""
        centers = list()
        for i in range(np.unique(bins).size):
            mask = np.where(bins == i)
            centx = 0.5 * (x[mask].max() + x[mask].min())
            centy = 0.5 * (y[mask].max() + y[mask].min())
            centers.append((centx, centy))
        return np.array(centers)

    def _nearby_mask(self, i, centers, x, y, bins):
        """Returns masks that flag all points within some distance of the
        i-th subvolume and also a mask that flags all point within the subvol.
        """
        # x-y coordinates of the ith subvolume center
        centx, centy = centers[i, :]
        # mask of all points within that subvolume
        binmask = np.where(bins == i)
        # distances from the center around which to look for points
        delta = 55
        mask_x = np.where(np.logical_and(x < centx + delta,
                                         x > centx - delta))[0]
        mask_y = np.where(np.logical_and(y < centy + delta,
                                         y > centy - delta))[0]
        # Mask that has all points both within and outside the subvol.
        around_inside_mask = np.intersect1d(mask_x, mask_y)
        # Mask that has only points outside the subvolumei
        around_mask = np.setxor1d(around_inside_mask, binmask)
        return around_mask, binmask

    def _average_cross_rr_pairs(self, bins, centers, x, y, z):
        """"Calculates the average number of RR pairs with one pair within the
        subvolume and one outside.
        """
        nsubs = np.unique(bins).size
        rrsubs = list()
        naverage = 10
        for i in np.random.choice(range(nsubs), size=naverage, replace=False):
            around_mask, binmask = self._nearby_mask(i, centers, x, y, bins)
            rrsubs.append(self._count_pairs(False, x[binmask], y[binmask],
                          z[binmask], x[around_mask], y[around_mask],
                          z[around_mask]))

        rrcross = rrsubs[0].copy()
        for i in range(rrcross.size):
            # First subvol already included
            for j in range(1, len(rrsubs)):
                rrcross[i]['npairs'] += (rrsubs[j])[i]['npairs']
            # After summing each subvol take the average and ensure int
            rrcross[i]['npairs'] = int(
                    round((rrcross[i])['npairs'] / len(rrsubs), 0))
        return rrcross

    def _Nmult(self, Nsamples):
        """Returns how many more uniform points to generate relative to
        number of simulated galaxies."""
        if Nsamples > 1e5:
            return 10
        elif Nsamples > 25000:
            return 30
        else:
            return 50

    def jackknife(self, samples):
        """Jackknifes the simulated galaxies within the simulation box.

        Parameters
        ----------
            samples : tuple
                (x, y, z) coordinates

        Returns
        ----------
            cov : numpy.ndarray
                Jackknife covariance matrix
        """
        nmult = self._Nmult(samples[0].size)
        # Recast samples into np.floa64 and unpack the sample
        x, y, z = [s.astype('float64') for s in samples]
        ndata = x.size
        nrand = int(ndata * nmult)
        # Generate randoms
        randx, randy, randz = self._randoms(nrand)
        # Assign points into bins along x-y
        bins = self._bins(x, y)
        randbins = self._bins(randx, randy)
        # Number of subvolumes
        nsubs = np.unique(bins).size
        # Calculate pairs over the whole box
        DD = self._count_pairs(True, x, y, z)
        DR = self._count_pairs(False, x, y, z, randx, randy, randz)
        RR = self._count_pairs(True, randx, randy, randz)

        # Estimate the average RR inside the subvolume
        RRsub_average, nrandsub_average = self._average_subvol_rr_pairs(
                randbins, randx, randy, randz)
        # Average number of RR pairs after removing one subvolume
        RRjack = self._subtract_pairs(RR.copy(), RRsub_average)
        # Estimate average cross border RR
        centers = self._centers(randx, randy, randbins)
        RRcross_average = self._average_cross_rr_pairs(randbins, centers,
                                                       randx, randy, randz)
        # From the RR counts after removing one subvolume remove the average
        # number of pairs that has one pair outside, weighted by 0.5
        RRjack = self._subtract_pairs(RRjack.copy(), RRcross_average,
                                      weight=0.5)
        # Now we begin the jackknifing process. However, here we already
        # precomputed the RR contribution and we know how many DD pairs are
        # in the whole box. So it is enough to count the DD and DR pairs
        # within each subvolume and subtract those from the total to obtain
        # the wp estimate
        wps = list()
        for i in range(nsubs):
            data_mask = np.where(bins == i)
            rand_mask = np.where(randbins == i)
            ndata_here = data_mask[0].size
            # ---------------------------#
            # Pairs within the subvolume #
            # ---------------------------#
            # If there are less than 3 galaxies in the subvolume do not
            # attempt to count the pairs
            if ndata_here < 3:
                DDjack = DD.copy()
                DRjack = DR.copy()
            else:
                # Count pairs within the subvolume
                DDin = self._count_pairs(True, x[data_mask], y[data_mask],
                                         z[data_mask])
                DRin = self._count_pairs(False, x[data_mask], y[data_mask],
                                         z[data_mask], randx[rand_mask],
                                         randy[rand_mask], randz[rand_mask])
                # Subtract the pairs we just counted from the global
                DDjack = self._subtract_pairs(DD.copy(), DDin)
                DRjack = self._subtract_pairs(DR.copy(), DRin)
            # -----------------------------#
            # Pairs crossing the subvolume #
            # -----------------------------#
            data_around_mask, __ = self._nearby_mask(i, centers, x, y, bins)
            rand_around_mask, __ = self._nearby_mask(i, centers, randx, randy,
                                                     randbins)
            DDcross = self._count_pairs(False, x[data_mask], y[data_mask],
                                        z[data_mask], x[data_around_mask],
                                        y[data_around_mask],
                                        z[data_around_mask])
            DRcross = self._count_pairs(False, x[data_mask], y[data_mask],
                                        z[data_mask], randx[rand_around_mask],
                                        randy[rand_around_mask],
                                        randz[rand_around_mask])

            DDjack = self._subtract_pairs(DDjack.copy(), DDcross, weight=0.5)
            DRjack = self._subtract_pairs(DRjack.copy(), DRcross, weight=0.5)

            wps.append(self._counts2wp(DDjack, DRjack, RRjack,
                                       ndata - ndata_here,
                                       nrand - nrandsub_average))

        covmat = np.cov(np.array(wps), rowvar=False, bias=True) * (nsubs - 1)
        return covmat
