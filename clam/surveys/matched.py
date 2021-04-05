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
"""A class for the matched catalog."""
import numpy as np

from astropy.cosmology import FlatLambdaCDM

from .base import BaseSurvey


class Matched(BaseSurvey):
    r"""The matched catalog. Parses the data."""
    def __init__(self, photometry):
        self.name = 'matched'
        self.redshift_range = (0.001, 0.05)
        self.apparent_magnitude_range = (10.0, 17.7)
        self.survey_area = 3750

        self.photometry = photometry

        self._parse_catalog()

    @property
    def data(self):
        return self._data

    def _parse_catalog(self):
        cat = np.load("/mnt/zfsusers/rstiskalek/pysham/data/"
                      "Matched_catalog_distance_corrected.npy")

        N = cat.size

        names = ['RA', 'DEC', 'Z', 'Kcorr', 'Mr', 'logMS', 'logMH', 'logMB',
                 'appMr', 'Veff', 'AGCNr']

        formats = ['float64'] * (len(names) - 1) + ['int64']
        data = np.zeros(N, dtype={'names': names, 'formats': formats})

        data['RA'] = cat['RA']
        data['DEC'] = cat['DEC']
        data['Z'] = cat['ZDIST']
        data['Kcorr'] = cat[self.photometry + '_KCORRECT']
        data['Mr'] = cat[self.photometry + '_ABSMAG']
        data['logMS'] = np.log10(cat[self.photometry + '_MASS'])
        # Assuming logMH is initially in h=0.7 units
        data['logMH'] = cat['logMH'] + 2 * np.log10(0.7)
        data['AGCNr'] = cat['AGCNr']

        data['logMB'] = np.log10(10**data['logMH'] + 1.4 * 10**data['logMS'])

        cosmo = FlatLambdaCDM(H0=100, Om0=0.295)
        dL = cosmo.luminosity_distance(data['Z']).value

        data['appMr'] = data['Mr'] + 5 * np.log10(dL) + 25 + data['Kcorr']

        m1 = data['Z'] > self.redshift_range[0]
        m2 = data['Z'] < self.redshift_range[1]
        m3 = data['appMr'] > self.apparent_magnitude_range[0]
        m4 = data['appMr'] < self.apparent_magnitude_range[1]

        m = m1
        for mask in [m2, m3, m4]:
            m = np.logical_and(m, mask)

        self._data = data[m]
