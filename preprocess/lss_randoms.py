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

"""Combines the small LSS random samples."""

import numpy
from astropy.io import fits
import os


def get_lss_randoms_from_files(folder):
    """
    Loads files with randoms and produces a single array. See [1] for more
    details. Must start with 'lss_random'.

    Parameters
    ----------
    folder : str
        Path to the folder holding the lss randoms.

    Returns
    -------
    RA : numpy.ndarray
        RA randoms
    DEC : numpy.ndarray
        DEC randoms

    References
    ----------
        .. [1] http://sdss.physics.nyu.edu/vagc/#download
    """

    RA = []
    DEC = []
    for f in os.listdir(folder):
        if f.startswith('lss_random'):
            rands = fits.open(folder + f)[1].data
            RA.append(rands['RA'])
            DEC.append(rands['DEC'])
    RA = numpy.hstack(RA)
    DEC = numpy.hstack(DEC)
    return RA, DEC


def main():
    folder = "../data/randoms/"
    RA, DEC = get_lss_randoms_from_files(folder)
    out = numpy.ndarray(RA.size, dtype={'names': ['RA', 'DEC'],
                                        'formats': [float, float]})
    out['RA'] = RA
    out['DEC'] = DEC

    print('Saving..')
    numpy.save('../data/lss_rands.npy', out)


if __name__ == '__main__':
    main()
