"""Model used to constain the galaxy-halo relation"""
# !/usr/bin/env python3
# coding: utf-8
from random import choice
from warnings import filterwarnings

import numpy as np

import tracemalloc

from time import time
import Corrfunc

from .jackknife import Jackknife
from .abundance_match import AbundanceMatch

filterwarnings("ignore", category=RuntimeWarning)


class Model(Jackknife):
    """Abundance matching model that does inference on the galaxy-halo
    relation. Contains functions to generate abundance matching mocks,
    calculate the clustering statistics within the mocks and compare it to
    the observations.

    Parameters:
    """
    # Return blobs
    _bounds = None
    _survey_wp = None

    def __init__(self, name, pars, scope, halo_proxy, nd_gal, halos, bounds,
                 survey_wp, boxsize, subside, rpbins, pimax, Njobs, Nmocks):
        # inherit jackknife
        super().__init__(boxsize, rpbins, pimax, subside, Njobs)
        # store inputs
        self.name = name
        self.pars = pars
        self.bounds = bounds
        self.survey_wp = survey_wp
        # define the AbundaceMatching class
        self.AM = AbundanceMatch(nd_gal, halos, scope, halo_proxy,
                                 Njobs, Nmocks, boxsize)

    @property
    def bounds(self):
        """Boundary for each parameter"""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        try:
            self._bounds = {p: Boundary(p, bounds[p])
                            for p in self.pars}
        except KeyError:
            raise ValueError('provide a boundary for each parameter')

    @property
    def survey_wp(self):
        """Precomputed survey projected two point correlation function.
        Returns the mean wp and the jackknife covariance matrix estimate."""
        return self._survey_wp['wp'], self._survey_wp['covmat']

    @survey_wp.setter
    def survey_wp(self, corrfunc):
        if not isinstance(corrfunc, dict):
            raise ValueError('must be a dictionary')
        if not all([p in list(corrfunc.keys()) for p in ['wp', 'covmat']]):
            raise ValueError('make sure survey CF dict has the right keys')
        self._survey_wp = corrfunc

    def _jackknife(self, sample):
        """Jackknife error estimate of the simulation. Depending on how many
        points are within the simultion picks how many times more randoms to
        use for jackknifing (always at least 10 times as many)
        """
        Nsamples = sample[0].size
        if Nsamples > 1e5:
            Nmult = 10
        elif Nsamples > 25000:
            Nmult = 30
        else:
            Nmult = 50
        return self.jackknife(sample, Nmult)

    def _stochastic(self, samples):
        """ Calculates the mean and covariance matrix for a list of catalogs.
        Assumes catalogs are independent for 1/N normalisation.
        Returns the mean CF and the covariance matrix
        """
        wps = np.array([(Corrfunc.theory.wp(boxsize=self.boxsize,
                                            nthreads=self.Njobs,
                                            binfile=self.rpbins,
                                            pimax=self.pimax, X=sample[0],
                                            Y=sample[1], Z=sample[2]))['wp']
                        for sample in samples])
        wp = np.mean(wps, axis=0)
        # Bias=True means normalisation by 1/N
        covmat = np.cov(wps, rowvar=False, bias=True)
        return wp, covmat

    def loglikelihood(self, theta):
        """Calculates the normal likelihood on the residues of the difference
        between the survey projected correlation function and the AM mock"""
        # Generate catalogs
        samples = self.AM.abundance_match(theta)
        wp_mean, stoch_covmat = self._stochastic(samples)
        # Randomly pick one AM mock to jackknife
        jack_covmat = self._jackknife(choice(samples))
        wp_survey, covmat_survey = self.survey_wp
        # Covariances are added
        covmat = stoch_covmat + jack_covmat + covmat_survey

        diff = (wp_mean - wp_survey).reshape(-1, 1)
        determinant = np.linalg.det(covmat)
        inverse = np.linalg.inv(covmat)
        chi2 = float(np.matmul(diff.T, np.matmul(inverse, diff)))

        blobs = {'wp': wp_mean, 'stoch_covmat': stoch_covmat,
                 'jack_covmat': jack_covmat}
        return - 0.5 * (np.log(determinant) + chi2), blobs

    def logprior(self, theta):
        """For now uniform prior over the parameter space"""
        if all([self.bounds[p].inside(theta[p]) for p in self.pars]):
            return 0.0
        return -np.infty

    def logposterior(self, theta):
        """Log posterior"""
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        start = time()
        logp = self.logprior(theta)
        if not np.isfinite(logp):
            return -np.infty
        ll, blobs = self.loglikelihood(theta)
        logpost = logp + ll
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print('Finished posterior call in {:.1f} sec'.format(time() - start))

        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
        return logpost, blobs


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
