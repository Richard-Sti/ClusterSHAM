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

"""Data selector working as a wrapper structured arrays."""

import warnings
import numpy

from .data_routines import (Routines, Conditions, Little_h)


class DataSelector:
    """
    Data selector serving as a wrapper around either a structured array or a
    .fits file. Applies preselection conditions, contains routines to either
    calculate properties not defined in the catalogue, and can convert
    units of little h.

    Internally calculations are always using the data cosmology (by extension
    `conditions` must be defined following the data cosmology). Outputs can
    be optionally converted to user cosmology.

    Parameters
    ----------
    frame : numpy.ndarray or astropy.io.fits.fitsrec.FITS_REC
        Structured array with survey data. In principle can be any structured
        array that returns numpy arrays.
    conditions : list, optional
        List of preselection conditions.
        Must be from `clustersham.survey.conditions`. By default `None`,
        meaning no preselection condition.
    routines : list, optional
        List of routines. Must be from `clustersham.survey.routines`. By
        default `None`, meaning no new routines.
    indices : dict, optional.
        Required for accessing `frame` attributes that are multi-dimensional
        arrays. Key must match the desired attribute's name and values are
        a dictionary with keys either `column` or `row`, such that both
        cannot be difed simultaneously. This then accesses either [:, `column`]
        or [`row`, :] of the array.
    little_h : list, optional
        List of little h transforms. Must be from
        `clustersham.survey.little_h`. By default `None`, meaning no
        transformation.
    """

    def __init__(self, frame, conditions=None, routines=None, indices=None,
                 little_h=None):
        self._frame = None
        self._mask = None
        self._routines = None
        self._indices = None
        self._little_h = None

        self.frame = frame
        self.routines = routines
        self.indices = indices
        self._mask = self._preselection(conditions)
        self.little_h = little_h

    @property
    def frame(self):
        """Data frame with the survey data."""
        return self._frame

    @frame.setter
    def frame(self, frame):
        """
        Sets frame. Also checks whether it is a structured array
        that supports indexing by attribute.
        """
        if frame.dtype.names is None:
            raise ValueError("`frame` must be a structured array.")
        self._frame = frame

    @property
    def routines(self):
        """Data selector's routines."""
        return self._routines

    @routines.setter
    def routines(self, routines):
        """Sets `routines`. Ensures only known routines are given."""
        if routines is None:
            return {}
        if not isinstance(routines, dict):
            raise ValueError("`routines` must be a dict.")
        # Check that conditions are well-defined
        for key, routine in routines.items():
            if not isinstance(key, str):
                raise ValueError("Routine's '{}' key must be a string. "
                                 "Currently {}.".format(routine, key))
            if routine.name  not in Routines.keys():
                raise ValueError("Invalid routine '{}'. Must be selected "
                                 "from `clustersham.utils.Routines"
                                 .format(routine.name))
        self._routines = routines

    @property
    def indices(self):
        """Row or column of 2D attribute's array."""
        return self._indices

    @indices.setter
    def indices(self, indices):
        """
        Sets `indices`. Checks each attribute only has either `row` or
        `column` specified.
        """
        if indices is None:
            self._indices = {}
        if not isinstance(indices, dict):
            raise ValueError("`indices` must be a dict.")

        err = ("Attribute '{}' can have at most a single key with integer "
               "value specified. This must be either `column` or `row`. "
               "Currently: '{}'")
        for key, indx in indices.items():
            if not isinstance(indx, dict):
                raise ValueError("Attribute '{}' index value must be a "
                                 "dictionary. Currently '{}'"
                                 .format(key, indx))

            if len(indx) > 1:
                raise ValueError(err.format(key, indx))
            # Now that we know indx only has one entry the logic simplifies
            indx_copy = indx.copy()
            col = indx_copy.pop('column', None)
            row = indx_copy.pop('row', None)
            if col is None and row is None:
                raise ValueError(err.format(key, indx))
            # Ta-da. Either col or row is not None. Mission accomplished.
        self._indices = indices

    @property
    def little_h(self):
        """The little h transformations."""
        return self._little_h

    @little_h.setter
    def little_h(self, transforms):
        """Sets ``little_h``. Checks there are known transforms."""
        if transforms is None:
            return {}
        if not isinstance(transforms, list):
            raise ValueError("`little_h` must be a list.")
        # Check that conditions are well-defined
        for transform in transforms:
            if transform.name  not in Little_h.keys():
                raise ValueError("Invalid little h transform '{}'. Must be "
                                 "selected from `clustersham.utils.Little_h"
                                 .format(routine.name))
        self._little_h = {transf.attr: transf for transf in transforms}

    def _preselection(self, conditions):
        """
        Calls individual conditions to determine which entires satisfy
        the preselection conditions.

        Parameters
        ----------
        conditions : list, optional
            List of preselection conditions.

        Returns
        -------
        mask : numpy.ndarray
            Mask of bools, which entries pass the preselection.
        """
        if conditions is None:
            return None
        if not isinstance(conditions, list):
            raise ValueError("`conditions` must be a list.")
        # Check that conditions are well-defined
        for condition in conditions:
            if condition.name  not in Conditions.keys():
                raise ValueError("Invalid condition '{}'. Must be selected "
                                 "from `clustersham.utils.Conditions"
                                 .format(condition.name))
        masks = [condition(self) for condition in conditions]
        if len(masks) == 1:
            return masks[0]
        # In case of multiple conditions do 'and'
        mask = masks[0]
        for m in masks[1:]:
            mask = numpy.logical_and(mask, m)
        return mask

    def _try_row_col(self, attr):
        """
        Called for `self.frame` attributes that are 2D arrays. Returns either
        the row or column, the other is returned as `None`.

        Parameters
        ----------
        attr : str
            Name of the attribute.

        Returns
        -------
        column : int
            Column index.
        row : int
            Row index
        """
        try:
            indx = self.indices[attr]
        except KeyError:
            raise ValueError("Attribute '{}' must have specified either a "
                             "column or a row in `indices`.".format(attr))
        # A doubly nested statement is not too terrible.. right
        try:
            return indx['column'], None
        except KeyError:
            try:
                return None, indx['row']
            except KeyError:
                raise RuntimeError("Unexpected behaviour. Exiting.")

    @staticmethod
    def _recast_array(X):
        """
        Recasts `X` into a more, ehm, well-behaved data type. Avoids
        `numpy.float32` etc.

        Parameters
        ----------
        X : numpy.ndarray
            Array to be recast into a nicer dtype.

        Returns
        -------
        X : numpy.ndarray
            Array that was recast into a nice dtype.
        """
        dtype = X.dtype
        if numpy.issubdtype(dtype, numpy.integer):
            return X.astype(int)
        elif numpy.issubdtype(dtype, numpy.inexact):
            return X.astype(float)
        elif numpy.issubdtype(dtype, numpy.bool_):
            return X.astype(bool)
        else:
            warnings.warn("Unknown dtype {}. Setting to 'float64'"
                          .format(dtype))
            return X.astype(float)

    def __getitem__(self, attr):
        """
        Returns `attr` from `self.frame`. Returned values pass preselection.

        Parameters
        ----------
        attr : str
            Attribute to be returned.
        """
        # Internally when routines request an attribute we do not want to
        # apply little h conversion to user cosmology
        if isinstance(attr, tuple):
            apply_transforms = attr[1]
            attr = attr[0]
        else:
            apply_transforms = True

        if attr in self.frame.dtype.names:
            if self.frame[attr].ndim == 1:
                out = self.frame[attr]
            elif self.frame[attr].ndim == 2:
                col, row = self._try_row_col(attr)
                if col is not None:
                    out = self.frame[attr][:, col]
                else:
                    out = self.frame[attr][row, :]
            else:
                raise ValueError("Attribute '{}' array of shape '{}' not "
                                 "supported. Can be at most a 2D array."
                                 .format(attr, self.frame[attr].shape))
            # Apply the preselection mask
            if self._mask is not None:
                out = out[self._mask]
            # Ensure a good dtype
            out = self._recast_array(out)
            # Possibly apply little h transform
            if apply_transforms and attr in self.little_h.keys():
                out = self.little_h[attr](out)
            return out

        elif attr in self.routines.keys():
            out = self.routines[attr](self)
            # Recast..
            out = self._recast_array(out)
            # Little h transform..
            if apply_transforms and attr in self.little_h.keys():
                out = self.little_h[attr](out)
            return out
        else:
            raise ValueError("Unknown attribute '{}', add a routine."
                             .format(attr))
