import numpy as np
from scipy.optimize import fminbound

import kmeans_radec
from astropy.cosmology import FlatLambdaCDM
from joblib import (Parallel, delayed, externals)


class NumberDensity(object):
    """Class used for computing the stelllar luminosity and mass functions
    for optical catalogs

    Parameters
    ----------
    survey : py:class `PySHAM.survey`
    """
    def __init__(self, survey, outfolder, nthreads=1):
        self.survey = survey
        self.RA, self.DEC, self.Z, self.Mr, self.logMS, self.Kcorr =\
            [survey.data[p] for p in ('RA', 'DEC', 'Z', 'Mr', 'logMS',
                                      'Kcorr')]
        self.cosmo = FlatLambdaCDM(H0=100, Om0=0.295)
        dL = self.cosmo.luminosity_distance(self.Z).value
        self.appMr = self.Mr + 25 + 5*np.log10(dL) + self.Kcorr

        self.Zmin, self.Zmax = self.Z.min(), self.Z.max()
        self.N = self.RA.size
        # fit a polynomial
        self.Kcorr_pars = np.polyfit(self.Z, self.Kcorr, deg=3)

        self.outfolder = outfolder
        self.nthreads = nthreads

    def _Kcorr_mean(self, x):
        """Returns values of the polynomial fit of Kcorr"""
        return np.polyval(self.Kcorr_pars, x)

    def _zmax_equation(self, zmax, M, Kcorr, Z, mlim):
        """Equation whose minimum is the best zmax value"""
        return np.abs(M + 25
                      + 5*np.log10(self.cosmo.luminosity_distance(zmax).value)
                      + Kcorr + (self._Kcorr_mean(zmax) - self._Kcorr_mean(Z))
                      - mlim)

    def _solve_zmax(self, i, mlim):
        """Zmax value solver passed into pools"""
        return fminbound(self._zmax_equation, self.Zmin, self.Zmax,
                         args=(self.Mr[i], self.Kcorr[i], self.Z[i], mlim))

    def zmax_zmin(self, mlim, nthreads):
        """Solver for either zmax or zmin depending mlim. (does a first order
        approximation of how Kcorrections evolve)"""
        with Parallel(n_jobs=nthreads, verbose=10, backend='loky') as par:
            out = par(delayed(self._solve_zmax)(i, mlim)
                      for i in range(self.N))
        externals.loky.get_reusable_executor().shutdown(wait=True)
        return np.array(out)

    def _vmax(self):
        """Calculates the maximum comoving volume"""
        mlim = self.survey.apparent_mr_range
        area = self.survey.survey_area
        # check if the area is in square degrees and convert
        if area > 4 * np.pi:
            area *= (np.pi/180)**2
        zmax = self.zmax_zmin(mlim[1], self.nthreads)
        zmin = self.zmax_zmin(mlim[0], self.nthreads)
        vmax = area / 3 * (self.cosmo.comoving_distance(zmax).value**3
                           - self.cosmo.comoving_distance(zmin).value**3)
        return vmax, zmax, zmin

    def _kmeans_labels(self, RA, DEC, ncents):
        """Returns kmeans labels for the given points"""
        X = np.vstack([RA, DEC]).T
        km = kmeans_radec.kmeans_sample(X, ncents, maxiter=250, tol=1.0e-5,
                                        verbose=0)
        return km.labels

    def number_density(self):
        ncents = 300
        labels = self._kmeans_labels(self.RA, self.DEC, ncents)
        vmax, zmax, zmin = self._vmax()
        nds = list()
        for feat, binwidth in zip((self.Mr, self.logMS), (0.2, 0.1)):
            # to reduce catalog edges effects exclude the brightest and
            # faintest 750 objects
            bins = np.arange(np.sort(feat)[750],
                             np.sort(feat)[feat.size - 750], binwidth)
            x = np.array([0.5*(bins[i+1] + bins[i])
                          for i in range(bins.size - 1)])

            y = list()
            # loop over the labels for jackknifing and mass/absmag bins
            for i in np.unique(labels):
                y_jack = np.zeros(bins.size - 1)
                for j in range(bins.size - 1):
                    mask = np.intersect1d(self._bin_mask(feat, (bins[j],
                                                                bins[j + 1])),
                                          np.where(labels != i))
                    y_jack[j] = (1/vmax[mask] / binwidth).sum()
                y.append(y_jack)

            y = np.array(y)
            nd = np.mean(y, axis=0)
            covmat = np.cov(y, rowvar=False, bias=True) * (ncents - 1)
            nd_std = np.sqrt(np.diagonal(covmat))
            nds.append(np.vstack([x, nd, nd_std]).T)

        for d, p in zip(nds, ('LF', 'MF')):
            np.save(self.outfolder + '{}.npy'.format(p), d)
        return nds, zmax, zmin

    def _bin_mask(self, feature, feature_range):
        return np.where(np.logical_and(feature > feature_range[0],
                                       feature < feature_range[1]))
