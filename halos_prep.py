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
"""Parses the darksky halo file and preps it for abundance matching."""
import argparse

import numpy as np

from astropy.cosmology import FlatLambdaCDM

from PySHAM.mocks import proxies


def main(proxy_name):
    print('Loading the halos.')
    halos = np.load('/mnt/zfsusers/rstiskalek/pysham/data/hlist_1.00000.npy')
    print('Loaded the halos.')

    proxy = proxies[proxy_name]()

    pos = ['x', 'y', 'z']
    pars = proxy.halos_parameters

    names = pos + pars
    formats = ['float64'] * len(names)
    out = np.zeros(halos.size, dtype={'names': names, 'formats': formats})

    for name in names:
        if proxy.name == 'vvir_proxy' and name == 'vvir':
            cosmology = FlatLambdaCDM(H0=68.8, Om0=0.295)
            out['vvir'] = proxy.vvir(halos, cosmology)
        elif proxy.name == 'zmpeak_proxy' and name == 'zmpeak':
            out['zmpeak'] = (1 / halos['mpeak_scale']) - 1
        else:
            out[name] = halos[name]
    fname = '/mnt/zfsusers/rstiskalek/pysham/data/halos_{}.npy'.format(
            proxy.name)
    np.save(fname, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", help="Catalog name", type=str)
    args = parser.parse_args()
    main(args.proxy)
