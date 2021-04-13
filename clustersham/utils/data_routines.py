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

"""Various routines for `.frame.DataSelector`."""

from abc import (ABC, abstractmethod)
import numpy


class BaseRoutine(ABC):
    """
    Base class for routines and conditions. Routines provide a recipe to
    provide new (galaxy) properties out of ones given in the catalogue and
    condition returns a mask of bools depending on whether the given property
    satifies some condition.

    The base class ensures that every routine and condition is callable.
    """

    @staticmethod
    def _check_attr(attr):
        """
        Checks that `attr` is a string.

        Parameters
        ----------
        attr : str
            Name of the attribute.

        Returns
        -------
        (attr, False) : tuple
            A tuple of the parameter and a False boolean. This ensures that
            the optional conversion to user-defined cosmology (different from
            data cosmology) is only done when returning results to the user.
        """
        if not isinstance(attr, str):
            raise ValueError("Parameter '{}' must be a string.".format(attr))
        return (attr, False)

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


class LogRoutine(BaseRoutine):
    """
    Routine returning base 10 logarithm of `selector[self.attr]`.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    """
    name = 'log_routine'

    def __init__(self, attr):
        self.attr = self._check_attr(attr)

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
    name = 'apparent_magnitude_routine'

    def __init__(self, redshift, absmag, cosmo, Kcorr=None):
        self.redshift = self._check_attr(redshift)
        self.absmag = self._check_attr(absmag)
        self.cosmo = cosmo
        if Kcorr is None:
            self.Kcorr = None
        else:
            self.Kcorr = self._check_attr(Kcorr)

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
            appmag += selector[self.Kcorr]
        return appmag


class ComovingDistanceRoutine(BaseRoutine):
    """
    Routine to calculate the comoving distance from redshift.

    Parameters
    ----------
    redshift : str
        Selector's redshift attribute.
    cosmo : `astropy.cosmology` object
        Cosmology of choice.
    """
    name = 'comoving_distance_routine'

    def __init__(self, redshift, cosmo):
        self.redshift = self._check_attr(redshift)
        self.cosmo = cosmo

    def __call__(self, selector):
        """
        Returns the comoving distance corresponding to selector's redshift.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        comoving_dist : numpy.ndarray
            Comoving distance in Mpc.
        """
        return self.cosmo.comoving_distance(selector[self.redshift]).value


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
    name = "finite_condition"

    def __init__(self, attr):
        self.attr = self._check_attr(attr)

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
        return numpy.isfinite(selector[self.attr])


class RangeCondition(BaseRoutine):
    """
    Range condition. Returns True for attribute's values that within
    `self.ext_range`.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    ext_range : len-2 tuple
        Attributes minimum and maximum values. To set the upper limit to
        positive infinity pass in string '+oo'. To set the lower limit to
        negative infinity pass in '-oo'.
    """
    name = "range_condition"

    def __init__(self, attr, ext_range):
        self.attr = self._check_attr(attr)
        # Tuple does not like assignment..
        if isinstance(ext_range, tuple):
            ext_range = list(ext_range)
        # Check if infinities passed in
        if isinstance(ext_range[0], str):
            if ext_range[0] == '-oo':
                ext_range[0] = -numpy.infty
            else:
                raise ValueError("Invalid lower limit string '{}'. Only "
                                 "allowed string is '-oo'."
                                 .format(ext_range[0]))
        if isinstance(ext_range[1], str):
            if ext_range[1] == '+oo':
                ext_range[1] = numpy.infty
            else:
                raise ValueError("Invalid upper limit string '{}'. Only "
                                 "allowed string is '+oo'."
                                 .format(ext_range[1]))

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


class IsEqualCondition(BaseRoutine):
    """
    Is equal condition. Returns True for values that are equal to `value`.

    Parameters
    ----------
    attr : str
        Selector's attribute to be log-transformed.
    value : bool, int, float, str
        Value to which selector's attribute values are compared to.
    is_equal : bool, optional
        Whether to do 'is_equal'. Alternatively will check 'not_equal'. By
        default True.
    """
    name = "is_equal_condition"

    def __init__(self, attr, value, is_equal=True):
        self.attr = self._check_attr(attr)
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError("`value` must be of either bool, int, float, "
                             "or str type. Currently '{}'"
                             .format(value.dtype))
        self.value = value
        if not isinstance(is_equal, bool):
            raise ValueError("`is_equal` must be either `True` or `False`.")
        self.is_equal = is_equal

    def __call__(self, selector):
        """
        Returns a boolean array whether the attribute values are equal to
        `self.value`.

        Parameters
        ----------
        selector : `clustersham.utils.DataSelector` object
            An object from which to access survey properties.

        Returns
        -------
        mask : numpy.ndarray
            Condition mask.
        """
        if not self.is_equal:
            return selector[self.attr] != self.value
        return selector[self.attr] == self.value


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
    name = "luminosity_log_mass_little_h"

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
    name = "absolute_magnitude_little_h"

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


# Dictionary with routine names
Routines = {LogRoutine.name: LogRoutine,
            ApparentMagnitudeRoutine.name: ApparentMagnitudeRoutine,
            ComovingDistanceRoutine.name: ComovingDistanceRoutine}


Conditions = {FiniteCondition.name: FiniteCondition,
              RangeCondition.name: RangeCondition,
              IsEqualCondition.name: IsEqualCondition}


Little_h = {LuminosityLogMassConvertor.name: LuminosityLogMassConvertor,
            AbsoluteMagnitudeConvertor.name: AbsoluteMagnitudeConvertor}
