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
"""Utility functions for other modules."""
import numpy as np

from joblib import (dump, load)


def dump_pickle(fname, obj):
    """Saves the ``obj`` to ``fname``."""
    with open(fname, 'wb') as handle:
        dump(obj, handle)


def load_pickle(fname):
    """Loads the object stored in ``fname``."""
    with open(fname, 'rb') as handle:
        data = load(handle)
    return data


def in_hull(point, hull, tolerance=1e-12):
    """Decides whether given ``point`` is in the scipy.spatial.ConvexHull
    object ``hull``."""
    """Returns True if poin in the hull."""
    eqs = hull.equations
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in eqs)
