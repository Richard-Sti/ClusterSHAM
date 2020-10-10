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
"""Possible halo proxies to be used for abundance matching."""

import numpy as np

from astropy import constants as const, units as u
from halotools import empirical_models

from .base import BaseProxy


class VirialMassProxy(BaseProxy):
    r"""Virial mass proxy for abundance matching defined as:

        .. math::
            m_{\alpha} = M_0 * \left M_{\mathrm{peak}} / M_0\right]^{\alpha},

    where :math:`M_0` and :math:`M_{\mathrm{peak}}` are the present and peak
    virial masses, respectively.
    """

    name = 'mvir_proxy'

    def __init__(self):
        self.halos_parameters = ['mvir', 'mpeak']

    def proxy(self, halos, theta):
        alpha = theta.pop('alpha')
        if theta:
            raise ValueError("Unrecognised parameters: {}"
                             .format(theta.keys()))
        proxy = halos['mvir'] * (halos['mpeak'] / halos['mvir'])**alpha
        proxy_mask = np.ones_like(proxy, dtype=bool)
        return proxy, proxy_mask


class ZmpeakVirialMassProxy(BaseProxy):
    r"""An extension of the virial mass proxy for abundance matching defined
    as:

        .. math::
            m_{\alpha} = M_0 * \left M_{\mathrm{peak}} / M_0\right]^{\alpha},

    where :math:`M_0` and :math:`M_{\mathrm{peak}}` are the present and peak
    virial masses, respectively. This proxy applies a 'zmpeak' cutoff, by
    only performing abundance matching on halos with 'zmpeak' earlier than
    'zcutoff'.
    """

    name = 'zmpeak_mvir_proxy'

    def __init__(self):
        self.halos_parameters = ['mvir', 'mpeak', 'zmpeak']

    def proxy(self, halos, theta):
        alpha = theta.pop('alpha')
        zcutoff = theta.pop('zcutoff')
        if theta:
            raise ValueError("Unrecognised parameters: {}"
                             .format(theta.keys()))

        proxy_mask = halos['zmpeak'] < zcutoff
        proxy = halos['mvir'] * (halos['mpeak'] / halos['mvir'])**alpha
        return proxy[proxy_mask], proxy_mask


class VirialVelocityProxy(BaseProxy):
    r"""Virial velocity proxy for abundance matching defined as

        .. math::
            v_{\alpha} = v_0 * \left[ v_{\max} / v_0\right]^{\alpha},

    where :math:`v_0` is the virial velocity evaluated at the peak halo mass
    and :math:`v_{\max}` is the maximum circular velocity.
    """

    name = 'vvir_proxy'

    def __init__(self):
        self.halos_parameters = ['vvir', 'Vmax@Mpeak']

    def proxy(self, halos, theta):
        alpha = theta.pop('alpha')
        if theta:
            raise ValueError("Unrecognised parameters: {}"
                             .format(theta.keys()))
        proxy = halos['vvir'] * (halos['Vmax@Mpeak'] / halos['vvir'])**alpha
        proxy_mask = np.ones_like(proxy, dtype=bool)
        return proxy, proxy_mask

    def vvir(self, halos, cosmology):
        """Calculates the virial velocity at peak halo mass as defined in [1].
        Note that following the probable definition in this work, the critical
        density here is taken at present time as well.

        Parameters
        ----------
            halos : numpy.ndarray
                Halos object
            cosmology : astropy.cosmology
                Cosmology used in the N-body simulation
        References
        ----------
        .. [1] Lehmann, Benjamin V et al. "The Concentration Dependence of the
               Galaxy-Halo Connection." arXiv:1510.05651

        Returns
        ----------
            vvir : numpy.ndarray
                Virial velocity in km/s
        """
        z_mpeak = (1.0 / halos['mpeak_scale']) - 1
        mvir = halos['mpeak']
        OmZ = cosmology.Om(z_mpeak)
        Delta_vir = empirical_models.delta_vir(cosmology, z_mpeak) / OmZ
        rho_crit = cosmology.critical_density0.to(u.kg/u.m**3)
        vvir = ((4 * np.pi / 3 * const.G**3 * Delta_vir * rho_crit)**(1/6)
                * (mvir * u.Msun.decompose())**(1/3)).to(u.km/u.s)
        return vvir.value
