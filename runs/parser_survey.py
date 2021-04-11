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

"""Parser to read the config file and generate the (paper) model."""

import numpy
import toml
from astropy.io import fits

# Later will delete this import
import sys
sys.path.append('../')

from clustersham.utils import (
        DataSelector, LogRoutine, ApparentMagnitudeRoutine, FiniteCondition,
        RangeCondition, IsEqualCondition, LuminosityLogMassConvertor,
        AbsoluteMagnitudeConvertor, Conditions, Routines, Little_h)

from astropy.cosmology import FlatLambdaCDM


Cosmologies = {'FlatLambdaCDM': FlatLambdaCDM}

class SurveyConfigParser:
    """
    Parses the survey config file. Returns `clustersham.utils.DataSelector`.

    Parameters
    ----------
        config_path : str
            Path to the toml config file.
    """
    def __init__(self, config_path):
        self.cnf = toml.load(config_path)

    def get_conditions(self):
        N = len(self.cnf['Conditions'])
        if  N == 0:
            return None

        conditions = [None] * N
        for i, (cond, kwargs) in enumerate(self.cnf['Conditions'].items()):
            if '_x_' in cond:
                cond = cond.split('_x_')[0]
            try:
                cond_out = Conditions[cond](**kwargs)
            except KeyError:
                raise ValueError("Unrecognised condition '{}'. Supported: "
                                 "'{}'. Supported delimiter is '_x_'."
                                 .format(cond, Conditions.keys()))
            conditions[i] = cond_out
        return conditions

    def get_routines(self):
        """Routines may depend on cosmology..."""
        N = len(self.cnf['Routines'])
        if  N == 0:
            return None

        routines = {}
        for routine_attr, routine_kwargs in self.cnf['Routines'].items():
            routine_kind = routine_kwargs.pop('kind')

            for key, cosmo_kwargs in routine_kwargs.items():
                if 'cosmo' in key:
                    cosmo = self.get_cosmology(cosmo_kwargs)
                    routine_kwargs.update({key: cosmo})

            try:
                routine = Routines[routine_kind](**routine_kwargs)
            except KeyError:
                raise ValueError("Unrecognised routine '{}'. Supported: "
                                 "'{}'.".format(routine_kind, Routines.keys()))
            routines.update({routine_attr: routine})
        return routines


    def get_cosmology(self, kwargs):
        kind = kwargs.pop('kind')
        try:
            return Cosmologies[kind](**kwargs)
        except KeyError:
            raise ValueError("Unrecognised cosmology '{}'. Consider adding it "
                             "to `Cosmologies`.".format(kind))

    def get_little_h(self):
        N = len(self.cnf['Little_h'])
        if  N == 0:
            return None

        transforms = [None] * N

        for i, (kind, kwargs) in enumerate(self.cnf['Little_h'].items()):
            # Get cosmologies
            for key, cosmo_kwargs in kwargs.items():
                if 'cosmo' in key:
                    cosmo = self.get_cosmology(cosmo_kwargs)
                    kwargs.update({key: cosmo})

            try:
                transforms[i] = Little_h[kind](**kwargs)
            except KeyError:
                raise ValueError("Unrecognised little h transform'{}'. "
                                 "Supported: '{}'."
                                 .format(kind, Little_h.keys()))
        return transforms

    def __call__(self):
        path = self.cnf['Main']['path']
        print(path)
        if path.endswith('.fits'):
            catalogue = fits.open(path)
            catalogue = catalogue[1].data
        elif path.endswith('.npy'):
            catalogue = numpy.load(path)
        else:
            raise ValueError("Unrecognised file format: {}".format(path))
        # Initialise the data selector
        conditions = self.get_conditions()
        routines = self.get_routines()
        indices = self.cnf['Indices']
        little_h = self.get_little_h()
        return DataSelector(catalogue, conditions, routines, indices,
                            little_h)

def main():
    parser = SurveyConfigParser('NSAsurvey.toml')
    selector = parser()
    print(selector['RA'])
    print(selector['RA'].size)

if __name__ == '__main__':
    main()
