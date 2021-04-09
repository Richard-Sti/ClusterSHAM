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

from abc import (ABC, abstractmethod)
import warnings
import numpy


class BaseRoutine(ABC):
    """
    Base class for routines and conditions. Routines provide a recipe to
    provide new (galaxy) properties out of ones given in the catalogue and
    condition returns a mask of bools depending on whether the given property
    satifies some condition.

    The base class ensures that every routine and condition is callable.
    """

    def _check_param(self, param):
        """
        Checks that `param` is a string.

        Parameters
        ----------
        param : str
            Name of the parameters

        Returns
        -------
        (param, False) : tuple
            A tuple of the parameter and a False boolean. This ensures that
            the optional conversion to user-defined cosmology (different from
            data cosmology) is only done when returning results to the user.
        """
        if not isinstance(param, str):
            raise ValueError("Parameter '{}' must be a string.")
        return (param, False)

    @abstractmethod
    def __call__(self, selector):
        """
        Returns the newly calculated property.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            And object from which to access survey properties.
        """
        pass


#
# =============================================================================
#
#                           Routines
#
# =============================================================================
#


class LogMassRoutine(BaseRoutine):
    """
    Routine returning base 10 logarithm of `selector[self.attr]`.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    """

    def __init__(self, attr):
        self.attr = self._check_param(attr)

    def __call__(self, selector):
        """
        Returns the transformed parameter.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        result : numpy.ndarray
            Transformed attribute.
        """
        return numpy.log10(selector[self.attr])

class ApparentMagnitudeRoutine(BaseRoutine):
    """
    Routine to calculate the apparent magnitude from redshift and absolute
    magnitude. Optionally may also specify K-correction attribute.

    Parameters
    ----------
    redshift : str
        Selector's redshift attribute.
    absmag : str
        Selectors absolute magnitude attribute.
    cosmo : `astropy.cosmology` object
        Data cosmology.
    Kcorr : str, optional
        Selector's K-correction attribute. By default `None`, no K-correction
        is applied.
    """

    def __init__(self, redshift, absmag, cosmo, Kcorr=None):
        self.redshift = self._check_param(redshift)
        self.absmag = self._check_param(absmag)
        self.cosmo = cosmology
        if Kcorr is None:
            self.Kcorr = None
        else:
            self.Kcorr = self._check_param(Kcorr)

    def __call__(self, selector):
        """
        Returns the apparent magnitude.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        appmag : numpy.ndarray
            Apparent magnitude.
        """
        # Make this selector.data_cosmo
        ldist = self.cosmo.luminosity_distance(selector[self.redshift])
        appmag = 25 + selector[self.absmag] + 5 * numpy.log10(ldist.value)
        if self.Kcorr is not None:
            appmag += selector[self.Kcorr_attr]
        return appmag


#
# =============================================================================
#
#                               Conditions 
#
# =============================================================================
#


class FiniteCondition(BaseRoutine):
    """
    Finite condition. Returns True for values that are finite. Typically
    serves to eliminate NaNs.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    """

    def __init__(self, attr):
        self.attr = self._check_param(attr)

    def __call__(self, selector):
        """
        Returns a boolean array whether the attribute values are finite.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        mask : numpy.ndarray
            Condition mask.
        """
        return numpy.isfinite(selector[self.param])


class RangeCondition(BaseRoutine):
    """
    Range condition. Returns True for attribute's values that within
    `self.ext_range`.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    ext_range : len-2 tuple
        Attributes minimum and maximum values.
    """

    def __init__(self, attr, ext_range):
        self.attr = self._check_param(attr)
        if ext_range[0] > ext_range[1]:
            ext_range = ext_range[::-1]
        self.ext_range = ext_range

    def __call__(self, selector):
        """
        Returns a boolean array whether the attribute values within
        `self.ext_range`.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        mask : numpy.ndarray
            Condition mask.
        """
        return numpy.logical_and(selector[self.attr] > self.ext_range[0],
                                 selector[self.attr] < self.ext_range[1])


class IsTrueCondition(BaseRoutine):
    """
    Is true condition. Returns True for values that are True or equal to 1.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    """

    def __init__(self, attr):
        self.attr = self._check_param(attr)

    def __call__(self, selector):
        """
        Returns a boolean array whether the attribute values are True.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        mask : numpy.ndarray
            Condition mask.
        """
        return selector[self.attr] == True


#
# =============================================================================
#
#                           Little h convertors 
#
# =============================================================================
#


class LuminosityLogMassConvertor:
    """
    A log-mass from scaled luminosity little h convertor. Transforms the
    logarithmic galaxy mass from survey's choice of little h to the user's
    choice of little h.

    Parameters
    ----------
    attr : str
        Selector's log-mass atribute to be "little h" converted.
    cosmo_data : `astropy.cosmology` object
        Data cosmology.
    cosmo_out : `astropy.cosmology` object
        User cosmology.
    """

    def __init__(self, attr, cosmo_data, cosmo_out):
        if not isinstance(attr, str):
            raise ValueError("Attribute '{}' must be of a str type"
                             .format(attr))
        self.attr = attr
        self.cosmo_data = cosmo_data
        self.cosmo_out = cosmo_out

    def __call__(self, X):
        """
        Rescales luminosity mass from the data cosmology to user cosmology.

        Parameters
        ----------
        X : numpy.ndarray
            Values to be converted.

        Returns
        -------
        result : numpy.ndarray
            Converted values.
        """
        H0_ratio = (self.cosmo_data.H0 / self.cosmo_out.H0).value
        return X + 2 * numpy.log10(H0_ratio)


class AbsoluteMagnitudeConvertor:
    """
    An absolute magnitude little h convertor. Transforms the absolute
    magnitude from survey's choice of little h to the user's choice of
    little h.

    Parameters
    ----------
    attr : str
        Selector's absolute magnitude atribute to be "little h" converted.
    cosmo_data : `astropy.cosmology` object
        Data cosmology.
    cosmo_out : `astropy.cosmology` object
        User cosmology.
    """

    def __init__(self, attr, cosmo_data, cosmo_out):
        if not isinstance(attr, str):
            raise ValueError("Attribute '{}' must be of a str type"
                             .format(attr))
        self.attr = attr
        self.cosmo_data = cosmo_data
        self.cosmo_out = cosmo_out

    def __call__(self, X):
        """
        Rescales absolute magnitude from the data cosmology to user cosmology.

        Parameters
        ----------
        X : numpy.ndarray
            Values to be converted.

        Returns
        -------
        result : numpy.ndarray
            Converted values.
        """
        H0_ratio = (self.cosmo_data.H0 / self.cosmo_out.H0).value
        return X - 5 * numpy.log10(H0_ratio)


#
# =============================================================================
#
#                               Selector 
#
# =============================================================================
#


class DataSelector:
    """Docs...."""

    def __init__(self, frame, conditions=None, routines=None, indices=None, little_h=None):
        self.frame = frame
        self.routines = routines

        self.little_h = {transf.param: transf for transf in little_h}


        self.indices = indices
        self._mask = None
        self._mask = self._preselection(conditions)


    def _preselection(self, conditions):
        if conditions is None:
            # Return mask of True
            return -1
        masks = [condition(self) for condition in conditions]
        if len(masks) == 1:
            return masks[0]
        mask = masks[0]
        for m in masks[1:]:
            mask = numpy.logical_and(mask, m)
        return mask

    def _try_row_col(self, attr):
        # Little nested try except never killed anybody, right:)
        try:
            indx = self.indices[attr]
        except KeyError:
            raise ValueError("Attribute '{}' must have specified either a "
                             "column or a row in `indices`.".format(attr))

        err = ("Attribute '{}' can have at most a single key with integer "
               "value specified. This must be either `column` or `row`. "
               "Currently: {}")

        if len(indx) > 1:
            raise ValueError(err.format(attr, indx))

        if 'column' in indx.keys() and 'row' not in indx.keys():
            return indx['column'], None
        elif 'column' not in indx.keys() and 'row' in indx.keys():
            return None, indx['row']
        else:
            raise ValueError(err.format(attr, indx))

    @staticmethod
    def _recast_array(X):
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
        """Change the __getitem__ call in the routines (sometimes)."""
        if isinstance(attr, tuple):
            apply_transforms = attr[1]
            attr = attr[0]
        else:
            apply_transforms = True

        if attr in self.frame.dtype.names:
            # Figure out which row/column to return..
            if self.frame[attr].ndim != 1:
                col, row = self._try_row_col(attr)

                if col is not None:
                    out = self.frame[attr][:, col]
                else:
                    out = self.frame[attr][row, :]

            else:
                out = self.frame[attr]
            # Apply mask and return
            if self._mask is not None:
                out = out[self._mask]

            out = self._recast_array(out)

            if apply_transforms and attr in self.little_h.keys():
                print('Transforming would like I and do..')
                out = self.little_h[attr](out)
#                out = ... call transform on out
            return out

        elif attr in self.routines.keys():
            out = self.routines[attr](self)
            out = self._recast_array(out)

            if apply_transforms and attr in self.little_h.keys():
                print('Transforming would like I and do..')
                out = self.little_h[attr](out)

            return out
        else:
            raise ValueError("Unknown attribute '{}', add a routine."
                             .format(attr))
