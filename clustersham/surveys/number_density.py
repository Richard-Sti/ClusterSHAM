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

"""Calculates the luminosity/mass function given Vmax."""

import numpy
from scipy.stats import binned_statistic


def galaxy_proxy_function(bins, proxy, vmax):
    """
    Calculates the galaxy proxy function (typically luminosity or stellar mass)
    using the non-parameteric :math:`V_\max` method.

    Parameters
    ----------
    proxy : numpy.ndarray
        Galaxy proxy.
    vmax : numpy.ndarray
        Maximum volume corresponding to the galaxy proxy.

    Returns
    -------
    res : structured numpy.ndarray
        Array with named fields `proxy`, `phi`, and `err`.
    """
    # Check that bins have equal spacing
    spacing = numpy.diff(bins)
    if numpy.unique(spacing).size != 1:
        raise ValueError("Bins must have uniform spacing. Currently: {}"
                         .format(spacing))
    dx = spacing[0]
    inv_vmax = 1. / vmax
    # Mean galaxy proxy in each bin
    x, __, __ =  binned_statistic(proxy, proxy, statistic='median', bins=bins)
    # Sum of inverse volumes in each bin
    phi, __, __ =  binned_statistic(proxy, inv_vmax, statistic='sum',
                                    bins=bins)
    # Counts in each bin
    bin_counts, __, __ =  binned_statistic(proxy, inv_vmax,
                                           statistic='count', bins=bins)
    # Divide phi by the bin-size
    phi /= dx
    err = phi / numpy.sqrt(bin_counts)
    res = numpy.ndarray(phi.size, dtype={'names': ['proxy', 'phi', 'err'],
                                         'formats': [float]*3})
    res['proxy'] = x
    res['phi'] = phi
    res['err'] = err
    return res
