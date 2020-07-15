"""Model used to constain the galaxy-halo relation"""
#!/usr/bin/env python3
# coding: utf-8
from random import choice
from warnings import filterwarnings

import numpy as np
from AbundanceMatching import (AbundanceFunction, add_scatter, rematch,
                               calc_number_densities, LF_SCATTER_MULT)
import Corrfunc

from .jackknife import Jackknife

filterwarnings("ignore", category=RuntimeWarning)

class Model(Jackknife):
    """Abundance matching model that does inference on the galaxy-halo
    relation. Contains functions to generate abundance matching mocks,
    calculate the clustering statistics within the mocks and compare it to
    the observations.

    Parameters:
    """
    # WRITE WHAT PARAMETERS
    # FINISH SETTING BOUNDS
    # WRITE THE PRIOR

    def __init__(self, name, scope, pars, boundaries, nd_gals, halos,
                 survey_corrfunc, nmocks, cosmology, boxsize, rpbins, pimax,
                 subside):

        super().__init__(boxsize, rpbins, pimax, subside)
        self._name = None
        self._scope = None
        self._pars = None
        self._boundaries = None
        self._cosmology = None
        self._nd_gals = None
        self._nmocks = None
        self._halos = None
        self._survey_corrfunc = None

        self.name = name
        self.scope = scope
        self.pars = pars
        self.boundaries = boundaries
        self.cosmology = cosmology
        self.nd_gals = nd_gals
        self.nmocks = nmocks

        self.survey_corrfunc = survey_corrfunc
        self.halos = halos

    @property
    def name(self):
        """Survey name"""
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError('provide a string')
        self._name = name

    @property
    def scope(self):
        """Scope defining the upper and lower catalog cut"""
        return self._scope

    @scope.setter
    def scope(self, scope):
        if not isinstance(scope, tuple):
            raise ValueError('provide a tuple')
        elif len(scope) != 2:
            raise ValueError('provide a tuple of length 2')
        elif not scope[0] < scope[1]:
            raise ValueError('min must be strictly less than max')
        self._scope = scope

    @property
    def pars(self):
        """List of parameters for the galaxy-halo relation"""
        return self._pars

    @property
    def boundaries(self):
        """Boundary for each parameter"""
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries):
        try:
            self._boundaries = {Boundary(p, boundaries[p]) for p in self.pars}
        except KeyError:
            raise ValueError('provide a boundary for each parameter')

    @pars.setter
    def pars(self, pars):
        if not isinstance(pars, list):
            raise ValueError('provide a list')
        if not all([isinstance(p, str) for p in pars]):
            raise ValueError('all pars must be strings')
        self._pars = pars

    @property
    def cosmology(self):
        """Astropy.cosmology object describing the halos simulation cosmology.
        """
        return self._cosmology

    @cosmology.setter
    def cosmology(self, cosmo):
        self._cosmology = cosmo

    @property
    def _am_type(self):
        """Abundance matching type. If LF then bright end first, if MF then
        faint end first"""
        if self.scope[0] > 0:
            return 'MF'
        return 'LF'

    @property
    def nd_gals(self):
        """Mass/luminosity function for the survey"""
        return self._nd_gals

    @nd_gals.setter
    def nd_gals(self, nd_gals):
        if not isinstance(nd_gals, np.ndarray):
            raise ValueError('provide a numpy array')
        elif nd_gals.shape[1] != 2:
            raise ValueError('must have only two columns')
        self._nd_gals = nd_gals

    @property
    def nmocks(self):
        """Number of abundance matching mocks to generate at each point of the
        posterior."""
        return self._nmocks

    @nmocks.setter
    def nmocks(self, nmocks):
        if not isinstance(nmocks, int):
            raise ValueError('provide an integer')
        self._nmocks = nmocks

    @property
    def survey_corrfunc(self):
        """Precomputed survey projected two point correlation function.
        Returns the mean wp and the jackknife covariance matrix estimate."""
        return self._survey_corrfunc['wp'], self._survey_corrfunc['covmat']

    @survey_corrfunc.setter
    def survey_corrfunc(self, corrfunc):
        if not isinstance(corrfunc, dict):
            raise ValueError('must be a dictionary')
        if not all([p in list(corrfunc.keys()) for p in ['wp', 'covmat']]):
            raise ValueError('make sure survey CF dict has the right keys')
        self._survey_corrfunc = corrfunc

    @property
    def halos(self):
        """Halos numpy structured array"""
        return self._halos

    @halos.setter
    def halos(self, halos):
        if not isinstance(halos, np.ndarray):
            raise ValueError('provide a numpy array')
        self._halos = halos


    def abundance_match(self, theta):
        """ Does abundance mathing on a halo list given to the class. If there
        is a need for different proxy I would recommend editing the code here
        to redefine the 'plist' variable accordingly.

        Parameters:
            theta : list; length two (alpha, scatter)
        """
        alpha, scatter = theta
        plist = self.halos['vvir']\
                * ((self.halos['vmax'] / self.halos['vvir'])**alpha)
        nd_halos = calc_number_densities(plist, self.boxsize)
        if self._am_type == 'LF':
            scatter *= LF_SCATTER_MULT
            am_f = AbundanceFunction(self.nd_gals[:, 0], self.nd_gals[:, 1],
                                     (-27.0, -15.0), faint_end_first=False)
        elif self._am_type == 'MF':
            am_f = AbundanceFunction(self.nd_gals[:, 0], self.nd_gals[:, 1],
                                     (7.5, 14.0), faint_end_first=True)
        # Deconvolute the scatter
        am_f.deconvolute(scatter, repeat=20)
        # Catalog with 0 scatter
        cat = am_f.match(nd_halos)
        cat_dec = am_f.match(nd_halos, scatter, False)
        # Start generating catalogs
        samples = list()
        for __ in range(self.nmocks):
            out = rematch(add_scatter(cat_dec, scatter), cat, am_f._x_flipped)
            # Eliminate NaNs and galaxies with mass/brightness below the cut
            mask = (~np.isnan(out)) & (out > self.scope[0])\
                    & (out < self.scope[1])
            samples.append([(self.halos[p])[mask] for p in ['x', 'y', 'z']])

        return samples

    def _jackknife(self, sample):
        """Jackknife error estimate of the simulation. Depending on how many
        points are within the simultion picks how many times more randoms to
        use for jackknifing (always at least 10 times as many)
        """
        nsamp = sample[0].size
        if nsamp > 1e5:
            nmult = 10
        elif nsamp > 25000:
            nmult = 30
        else:
            nmult = 50
        return self.jackknife(sample, nmult)

    def _stochastic(self, samples):
        """ Calculates the mean and covariance matrix for a list of catalogs.
        Assumes catalogs are independent for 1/N normalisation.
        Returns the mean CF and the covariance matrix
        """
        wps = np.array([
            (Corrfunc.theory.wp(boxsize=self.boxsize, nthreads=1,\
            binfile=self.rpbins, pimax=self.pimax, X=sample[0],\
            Y=sample[1], Z=sample[2]))['wp']\
            for sample in samples])
        wp = np.mean(wps, axis=0)
        # Bias=True means normalisation by 1/N
        covmat = np.cov(wps, rowvar=False, bias=True)
        return wp, covmat

    def loglikelihood(self, theta):
        """Calculates the normal likelihood on the residues of the difference
        between the survey projected correlation function and the AM mock"""
        # Generate catalogs
        samples = self.abundance_match(theta)
        wp_mean, stoch_covmat = self._stochastic(samples)
        # Randomly pick one AM mock to jackknife
        jack_covmat = self._jackknife(choice(samples))
        wp_survey, covmat_survey = self.survey_corrfunc
        # Covariances are added
        covmat = stoch_covmat + jack_covmat + covmat_survey

        diff = (wp_mean - wp_survey).reshape(-1, 1)
        determinant = np.linalg.det(covmat)
        inverse = np.linalg.inv(covmat)
        chi2 = float(np.matmul(diff.T, np.matmul(inverse, diff)))

        return - 0.5 * (np.log(determinant) - chi2)


    def logprior(self, theta):
        """For now uniform prior over the parameter space"""
        if all([self.boundaries[p].inside(theta[p]) for p in self.pars]):
            return 0.0
        return -np.infty


    def logposterior(self, theta):
        """Log posterior"""
        logp = self.logprior(theta)
        if not np.isfinite(logp):
            return -np.infty
        return logp + self.loglikelihood(theta)

class Boundary(object):
    """Length 2 tuple that has min and max and checks whether a point
    is withing the boundary
    -----------------------
    Parameters:
        name: str
            Parameter name
        boundary: length-2 tuple
            Boundaries
    -----------------------
    Attributes:
        min
    """
    def __init__(self, name, boundary):
        self._name = None
        self._boundary = None

        self.name = name
        self.boundary = boundary

    @property
    def name(self):
        """Parameter name"""
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError('provide a str')
        self._name = name

    @property
    def boundary(self):
        """Tuple representing the boundary (inclusive)"""
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if not isinstance(boundary, tuple):
            raise ValueError('must be a tuple')
        elif len(boundary) != 2:
            raise ValueError('must be a tuple of length 2')
        elif not boundary[1] > boundary[0]:
            raise ValueError('max must be strictly larger than min')
        self._boundary = boundary

    @property
    def min(self):
        """Boundary min"""
        return self._boundary[0]

    @property
    def max(self):
        """Boundary max"""
        return self._boundary[1]

    @property
    def width(self):
        """Boundary width"""
        return self._boundary[1] - self._boundary[0]

    def inside(self, point):
        """Checks if the point is within the boundary, returns True/False"""
        return bool(self.min <= point <= self.max)
