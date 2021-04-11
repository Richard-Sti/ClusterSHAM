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

"""Parser to read surveys and return data selectors."""

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
        sub : str, optional
            Subsample unique ID as defined by the config file. By default
            does not apply any cuts
    """
    def __init__(self, config_path, sub=None):
        self.cnf = toml.load(config_path)
        self.sub = sub
        self.cut_condition = None

    def get_conditions(self):
        """
        Loads possible conditions from `clustersham.utils.Conditions`
        """
        # If not condition is specified exit
        N = len(self.cnf['Conditions'])
        if N == 0:
            return None
        # Load the individual conditions, '_x_' is used as delimiter to
        # distinguish multiple conditions of the same type
        # We will be appending as we cannot know the size in advance
        conditions = []
        for i, (cond, kwargs) in enumerate(self.cnf['Conditions'].items()):
            # Check if this is the subsample condition. Delimiter is _SUB_
            if '_SUB_' in cond:
                if self.sub is not None and self.sub in cond:
                    cond_split = cond.split('_SUB_')[0]
                else:
                    continue
            elif '_x_' in cond:
                cond_split = cond.split('_x_')[0]
            else:
                cond_split = cond
            try:
                cond_out = Conditions[cond_split](**kwargs)
            except KeyError:
                raise ValueError("Unrecognised condition '{}'. Supported: "
                                 "'{}'. Supported delimiter is '_x_' or _SUB_ "
                                 "to distinguish subsamples."
                                 .format(cond, Conditions.keys()))
            conditions.append(cond_out)
            # Cache the subsample condition cut
            if '_SUB_' in cond:
                self.cut_condition = cond_out
        return conditions

    def get_routines(self):
        """
        Loads possible routines from `clustersham.utils.Routines`. Routines
        may depend on Astropy's cosmology, hence these are loaded here too.
        """
        # Exit if no routine
        N = len(self.cnf['Routines'])
        if N == 0:
            return None
        # Go over the routines
        routines = {}
        for routine_attr, routine_kwargs in self.cnf['Routines'].items():
            routine_kind = routine_kwargs.pop('kind')
            # Check if this is a cosmology. Must contain 'cosmo' in name
            for key, cosmo_kwargs in routine_kwargs.items():
                if 'cosmo' in key:
                    cosmo = self.get_cosmology(cosmo_kwargs)
                    routine_kwargs.update({key: cosmo})
            # Check if a valid Routine
            try:
                routine = Routines[routine_kind](**routine_kwargs)
            except KeyError:
                raise ValueError("Unrecognised routine '{}'. Supported: "
                                 "'{}'.".format(routine_kind, Routines.keys()))
            routines.update({routine_attr: routine})
        return routines

    def get_cosmology(self, kwargs):
        """
        A special parser for Astropy's cosmology objects as this is required
        by both routines and little_h transforms.
        """
        kind = kwargs.pop('kind')
        try:
            return Cosmologies[kind](**kwargs)
        except KeyError:
            raise ValueError("Unrecognised cosmology '{}'. Consider adding it "
                             "to `Cosmologies`.".format(kind))

    def get_little_h(self):
        """
        Loads little h transforms. May depend on cosmology.
        """
        # Exit gracefully if no little h transform
        N = len(self.cnf['Little_h'])
        if N == 0:
            return None

        transforms = [None] * N
        for i, (kind, kwargs) in enumerate(self.cnf['Little_h'].items()):
            # Get cosmologies
            for key, cosmo_kwargs in kwargs.items():
                if 'cosmo' in key:
                    cosmo = self.get_cosmology(cosmo_kwargs)
                    kwargs.update({key: cosmo})
            # Check if a valid transform
            try:
                transforms[i] = Little_h[kind](**kwargs)
            except KeyError:
                raise ValueError("Unrecognised little h transform'{}'. "
                                 "Supported: '{}'."
                                 .format(kind, Little_h.keys()))
        return transforms

    def __call__(self):
        """
        Goes through conditions, routines, little h transforms and returns
        `DataSelector`.
        """
        path = self.cnf['Main']['path']
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
