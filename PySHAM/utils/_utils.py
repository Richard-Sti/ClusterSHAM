import numpy as np
from joblib import (dump, load)
from astropy import constants as const, units as u
from astropy.cosmology import FlatLambdaCDM
from halotools import empirical_models


def dump_pickle(fname, obj):
    with open(fname, 'wb') as handle:
        dump(obj, handle)


def load_pickle(fname):
    with open(fname, 'rb') as handle:
        data = load(handle)
    return data


def prep_halos(halos_path, tags):
    print('loading')
    halos = np.load(halos_path)
    print('loaded')

    names = ['x', 'y', 'z'] + [tag for tag in tags]
    formats = ['float64'] * len(names)
    N = halos['x'].size
    out = np.zeros(N, dtype={'names': names, 'formats': formats})
    for name in names:
        if not name == 'vvir':
            out[name] = halos[name]
        else:
            cosmology = FlatLambdaCDM(H0=68.8, Om0=0.295)
            print('doing vvir')
            vvir = vvir_lehmann(halos, cosmology)
            print('done vvir')
            out[name] = vvir
    return out


def vvir_lehmann(halos, cosmology):
    """Calculates virial velocity in km/s defined following
    Lehmann et al 2015

    -------------------------------------------------
    Parameters:
        halos: np.ndarray
            Halos object
        cosmology: astropy.cosmology object
            Cosmology used in the N-body simulation
    -------------------------------------------------
    Returns:
        vvir: np.ndarray
            Velocity in km/s
    """
    z_mpeak = (1.0 / halos['mpeak_scale']) - 1
    mvir = halos['mpeak']
    OmZ = cosmology.Om(z_mpeak)
    Delta_vir = empirical_models.delta_vir(cosmology, z_mpeak) / OmZ
    rho_crit = cosmology.critical_density0.to(u.kg/u.m**3)
    vvir = ((4 * np.pi / 3 * const.G**3 * Delta_vir * rho_crit)**(1/6)
            * (mvir * u.Msun.decompose())**(1/3)).to(u.km/u.s)
    return vvir.value


def vvir_proxy(halos, **kwargs):
    alpha = kwargs['alpha']
    return halos['vvir'] * (halos['Vmax@Mpeak']/halos['vvir'])**alpha


def mvir_proxy(halos, **kwargs):
    alpha = kwargs['alpha']
    return halos['mvir'] * (halos['mpeak'] / halos['mvir'])**alpha


def in_hull(point, hull, tolerance=1e-12):
    """Returns True if poin in the hull"""
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
               for eq in hull.equations)
