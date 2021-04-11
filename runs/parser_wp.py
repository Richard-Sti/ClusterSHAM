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

import numpy
import toml

import sys
sys.path.append('../')

from clustersham.surveys import Projected2PointCorrelation

class Projected2PointCorrelationParser:

    def __init__(self, config_path):
        self.cnf = toml.load(config_path)



    def __call__(self):
        # Get rpbins
        pars = ['rpmin', 'rpmax', 'nrpbins']
        args = [self.cnf['Correlator'].pop(key, None) for key in pars]
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError("'{}' must be specified in 'Correlato''"
                                 .format(pars[i]))
        rpbins = numpy.logspace(numpy.log10(args[0]), numpy.log10(args[1]),
                                args[2] + 1)
        kwargs = self.cnf['Correlator']
        kwargs.update({'rpbins': rpbins})
        return Projected2PointCorrelation(**kwargs)

if __name__ == '__main__':
    parser = Projected2PointCorrelationParser('NSAconfig.toml')
    model = parser()
    print(model)
    print(model.rpbins)

