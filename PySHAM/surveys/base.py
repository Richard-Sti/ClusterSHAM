# Copyright (C) 2020  Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.  #
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""A base class for all surveys."""

from abc import (ABCMeta, abstractmethod)
from six import add_metaclass

import numpy as np
from scipy.constants import speed_of_light

from ..mocks.base import Base


@add_metaclass(ABCMeta)
class BaseSurvey(object):
    r"""A base model to hold properties of different surveys that are being
    analysed.

    Attributes
    ----------
    name
    redshift_range
    apparent_magnitude_range
    survey_area
    data
    """
    _name = None
    _redshift_range = None
    _apparent_magnitude_range = None
    _survey_area = None
    _data = None

    @property
    def name(self):
        """Returns the name of the survey."""
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of the survey."""
        if not isinstance(name, str):
            raise ValueError("``name`` must be a string.")
        self._name = name

    @property
    def redshift_range(self):
        """Returns the specified redshift range."""
        return self._redshift_range

    @redshift_range.setter
    def redshift_range(self, zrange):
        """Sets the redshift range."""
        if not isinstance(zrange, tuple):
            raise ValueError("``redshift_range`` must be a tuple.")
        if not len(zrange) == 2:
            raise ValueError("``redshift_range`` must be a tuple of len-2")
        self._redshift_range = zrange

    @property
    def apparent_magnitude_range(self):
        """Return the apparent magnitude range."""
        return self._apparent_magnitude_range

    @apparent_magnitude_range.setter
    def apparent_magnitude_range(self, apprange):
        """Sets the apparent magnitude range."""
        if not isinstance(apprange, tuple) or not len(apprange) == 2:
            raise ValueError("``apparent_magnitude_range`` must "
                             "be a len-2 tuple")
        self._apparent_magnitude_range = apprange

    @property
    def survey_area(self):
        """Returns the survey area in degrees."""
        return self._survey_area

    @survey_area.setter
    def survey_area(self, area):
        """Sets the survey area."""
        if not isinstance(area, (int, float)):
            raise ValueError("``survey_area`` must be an int.")
        self._survey_area = float(area)

    @property
    @abstractmethod
    def data(self):
        """Returns the parsed catalog data."""
        pass

    @abstractmethod
    def _parse_catalog(self):
        """Parses the catalog, applying chosen cuts and potentially
        eliminating any outliers.
        """
        pass

    def scopes(self, handle, faint_end_first, fraction_bins=None):
        """Finds scopes corresponding to ``fraction_bins``. By default

            ``fraction_bins`` = [0.0, 0.015, 0.15, 0.4, 0.9],

        where objects are ranked from brightest to faintest.

        Parameters:
        ----------
            handle: str
                Galaxy property (``Mr``, ``logMS``, ...). Must be defined
                in ``self.data``.
            faint_end_first: bool
                Whether faint end is first. ``True`` for mass and ``False``
                for magnitudes.
            fraction_bins: list
                Described above.
        """

        if fraction_bins is None:
            fraction_bins = [0.0, 0.015, 0.15, 0.4, 0.9]
        feature = np.sort(self.data[handle])
        if faint_end_first:
            feature = np.flip(feature, axis=0)
        cuts = [feature[int(x * feature.size)] for x in fraction_bins]
        # make the scope tuples
        return [(cuts[i], cuts[i+1]) for i in range(len(cuts) - 1)]

    def scope_selection(self, scope, handle):
        """Returns galaxies corresponding to the particular scope.

        Parameters
        ----------
            scope: tuple
                Upper and lower limit on ``handle``.
            handle: str
                Galaxy property (``Mr``, ``logMS``, ...). Must be defined
                in ``self.data``.
        """
        a, b = scope
        if not b > a:
            scope = (b, a)
        mask = np.logical_and(self.data[handle] > scope[0],
                              self.data[handle] < scope[1])
        return self.data[mask]

    def handle(self, nd_type):
        """Returns the handle corresponding to ``nd_type``."""
        if nd_type == 'SMF':
            return 'logMS'
        elif nd_type == 'LF':
            return 'Mr'
        elif nd_type == 'BMF':
            return 'logMB'
        elif nd_type == 'HIMF':
            return 'logMH'
        else:
            raise ValueError("Unrecognised ``nd_type`` {}".format(nd_type))

    def faint_end_first(self, handle):
        """Whether the minimum value of ``handle`` coresponds to the
        brightest/heavist sample.
        """
        if handle == 'Mr':
            return False
        elif handle == 'logMS':
            return True
        elif handle == 'logMB':
            return True
        elif handle == 'logMH':
            return True
        else:
            raise ValueError("Unknown ``handle`` {}.".format(handle))

    def cutoffs(self, handle):
        """Returns the faintest/least massive or brightest/heavist object
        for ``handle`` + some tolerance for extrapolating."""
        data = self.data[handle]
        xmin = np.min(data)
        xmax = np.max(data)
        if handle == 'Mr':
            xmin += 2.5
        else:
            xmin -= 1.0
        return xmin, xmax


@add_metaclass(ABCMeta)
class BaseProjectedCorrelationFunction(Base):
    r"""A base model for PySHAM.survey.ProjectedCorrelationFunction. Handles
    primarily inputs.

    Attributes
    ----------
    data
    randoms
    Njack
    Nmult
    """
    _data = None
    _randoms = None
    _Njack = None
    _Nmult = None

    @property
    def data(self):
        """Returns survey data. This should already correspond to the given
        scope that is being investigated.
        """
        return self._data

    @data.setter
    def data(self, data):
        """Sets ``self.data``."""
        if not isinstance(data, np.ndarray):
            raise ValueError("``data`` must be of numpy.ndarray type.")
        # Replaces redshift (Z) in ``data`` with CZ in km/s
        CZ = data['Z'] * speed_of_light * 1e-3

        names = list(data.dtype.names) + ['CZ', 'weights', 'labels']
        names.remove('Z')
        formats = ['float64'] * len(names[:-1]) + ['int64']
        N = CZ.size
        # make a new array
        self._data = np.zeros(N, dtype={'names': names, 'formats': formats})
        for p in data.dtype.names:
            if p == 'Z':
                self._data['CZ'] = CZ
            else:
                self._data[p] = data[p]

    @property
    def randoms(self):
        """Returns the random sky position samples over the survey angular
        distribution.
        """
        return self._randoms

    @randoms.setter
    def randoms(self, randoms):
        """Sets the ``randoms`` and assigns them ``CZ``.
        """
        if not isinstance(randoms, np.ndarray):
            raise ValueError("``randoms_path`` must be numpy.ndaarray.")
        N = self.data.size * self.Nmult
        randoms = randoms[:N]
        CZ = np.random.choice(self.data['CZ'], N, replace=True)
        names = ['RA', 'DEC', 'CZ', 'weights', 'labels']
        formats = ['float64'] * len(names[:-1]) + ['int64']
        self._randoms = np.zeros(N, dtype={'names': names, 'formats': formats})
        for name in ['RA', 'DEC']:
            self._randoms[name] = randoms[name]
        self._randoms['CZ'] = CZ

    @property
    def Njack(self):
        """Returns the number of jackknifes."""
        return self._Njack

    @Njack.setter
    def Njack(self, Njack):
        """Sets the number of jackknifes."""
        if not isinstance(Njack, int):
            raise ValueError("``Njack`` must be an int.")
        self._Njack = Njack

    @property
    def Nmult(self):
        """Returns how many times more randoms used to relative to data."""
        return self._Nmult

    @Nmult.setter
    def Nmult(self, Nmult):
        """Sets the ``Nmult``."""
        if not isinstance(Nmult, int):
            raise ValueError("``Nmult`` must be an int.")
        self._Nmult = Nmult
