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

"""Parser for Projected2PointCorrelation."""

import numpy
import toml

import sys
sys.path.append('../')

from clustersham.surveys import Projected2PointCorrelation


class Projected2PointCorrelationParser:
    """
    A parser for `clustersham.surveys.Projected2PointCorrelation`.

    Parameters
    ----------
        config_path : str
            Path to the toml config file.
    """

    def __init__(self, config_path):
        self.cnf = toml.load(config_path)

    def __call__(self):
        r"""
        Gets the log-spaced :math:`r_p` bins and returns
        `clustersham.utils.Projected2PointCorrelation`.
        """
        pars = ['rpmin', 'rpmax', 'nrpbins']
        args = [self.cnf['Correlator'].pop(key, None) for key in pars]
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError("'{}' must be specified in 'Correlator'"
                                 .format(pars[i]))
        rpbins = numpy.logspace(numpy.log10(args[0]), numpy.log10(args[1]),
                                args[2] + 1)
        kwargs = self.cnf['Correlator']
        kwargs.update({'rpbins': rpbins})
        return Projected2PointCorrelation(**kwargs)
