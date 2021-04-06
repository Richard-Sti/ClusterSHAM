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
"""Classes for NYU and NSA catalogs."""

import numpy as np

from healpy.rotator import angdist

from .base import BaseSurvey


class NYU(BaseSurvey):
    r"""NYU catalog. Parses the raw NYU data for which apparent magnitudes and
    apparent magnitudes were precomputed.
    """
    def __init__(self):
        self.name = 'NYU'
        self.redshift_range = (0.01, 0.15)
        self.apparent_magnitude_range = (10.0, 17.7)
        self.survey_area = 7100

        self._parse_catalog()

    @property
    def data(self):
        return self._data

    def _parse_catalog(self):
        path = "/mnt/zfsusers/rstiskalek/pysham/data/NYUcatalog_wpols.npy"
        cat = np.load(path)
        try:
            inpolygon = cat['IN_POL']
        except ValueError:
            raise ValueError('must provide ``IN_POL`` key.')
        N = cat['RA'][inpolygon].size
        names = ['Mr', 'apMr', 'Kcorr', 'MS', 'RA', 'DEC', 'Z', 'dist']
        formats = ['float64'] * len(names)
        data = np.zeros(N, dtype={'names': names, 'formats': formats})
        # create the strucured array
        for name in names:
            try:
                data[name] = cat[name][inpolygon]
            except ValueError:
                raise ValueError("Input catalog missing ``{}``.".format(name))
        # remove outliers
        masks = NSA_NYU_outliers(data['RA'], data['DEC'])
        scopes = [self.apparent_magnitude_range, self.redshift_range]
        for p, scope in zip(['apMr', 'Z'], scopes):
            mask = np.logical_and(data[p] > scope[0], data[p] < scope[1])
            masks.append(mask)
        # check MS and Mr are finite
        for p in ['MS', 'Mr']:
            mask = np.isfinite(data[p])
            masks.append(mask)

        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = np.logical_and(final_mask, mask)
        # apply the selection
        data = data[final_mask]
        # Replace MS with logMS
        data['MS'] = np.log10(data['MS'])
        names = list(data.dtype.names)
        i = None
        for j in range(len(names)):
            if names[j] == 'MS':
                i = j
        names[i] = 'logMS'
        data.dtype.names = names
        self._data = data


class NSA(BaseSurvey):
    r"""NSA catalog. Parses the raw NYU data for which apparent magnitudes and
    apparent magnitudes were precomputed.

    Parameters
    ----------
    photometry: str
        Can be either 'SERSIC' or 'ELPETRO'.
    """
    _photometry = None

    def __init__(self, photometry):
        self.name = 'NSA'
        self.redshift_range = (0.01, 0.15)
        self.apparent_magnitude_range = (10.0, 17.6)
        self.survey_area = 7100

        self.photometry = photometry
        self._parse_catalog()

    @property
    def photometry(self):
        """Returns the selected photometry."""
        return self._photometry

    @photometry.setter
    def photometry(self, photometry):
        """Sets the photometry."""
        if not isinstance(photometry, str):
            raise ValueError("``photometry`` must be str.")
        if photometry not in ['SERSIC', 'ELPETRO']:
            raise ValueError("``photometry`` must be ``SERSIC`` or "
                             "``ELPETRO``")
        self._photometry = photometry

    @property
    def data(self):
        return self._data

    def _parse_catalog(self):
        path = "/mnt/zfsusers/rstiskalek/pysham/data/NSAcatalog_wpols.npy"
        cat = np.load(path)
        try:
            inpolygon = cat['IN_POL']
        except ValueError:
            raise ValueError('must provide ``IN_POL`` key.')

        N = cat['RA'][inpolygon].size
        pnames = ['Mr', 'apMr', 'Kcorr', 'MS']
        gnames = ['RA', 'DEC', 'Z', 'dist']

        names = pnames + gnames
        formats = ['float64'] * len(names)
        data = np.zeros(N, dtype={'names': names, 'formats': formats})
        # create the structured array
        for pname, gname in zip(pnames, gnames):
            try:
                data[gname] = cat[gname][inpolygon]
                data[pname] = cat[self.photometry + '_' + pname][inpolygon]
            except ValueError:
                raise ValueError("Input catalog missing ``{}`` or ``{}``."
                                 .format(pname, gname))
        # remove outliers
        masks = NSA_NYU_outliers(data['RA'], data['DEC'])
        scopes = [self.apparent_magnitude_range, self.redshift_range]
        for p, scope in zip(['apMr', 'Z'], scopes):
            mask = np.logical_and(data[p] > scope[0], data[p] < scope[1])
            masks.append(mask)
        # check MS and Mr are finite
        for p in ['MS', 'Mr']:
            mask = np.isfinite(data[p])
            masks.append(mask)

        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = np.logical_and(final_mask, mask)
        # apply the selection
        data = data[final_mask]
        # Replace MS with logMS
        data['MS'] = np.log10(data['MS'])
        names = list(data.dtype.names)
        i = None
        for j in range(len(names)):
            if names[j] == 'MS':
                i = j
        names[i] = 'logMS'
        data.dtype.names = names
        self._data = data


def NSA_NYU_outliers(RA, dec):
    """Used for NSA and NYU to by-hand remove some outlier galaxies.
    RA and dec must be in degrees.
    """
    # Eliminate some outlier galaxies
    X = np.vstack([np.pi/2 - np.deg2rad(dec), np.deg2rad(RA)])
    masks = [None] * 3

    masks[0] = 0.15 < angdist(X, [0.16*np.pi, 1.44*np.pi])

    masks[1] = 0.075 < angdist(X, [0.51*np.pi, 1.38*np.pi])
    masks[2] = 0.04 < angdist(X, [np.pi/2, 0.7*np.pi])
    return masks
