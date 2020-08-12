import numpy as np

from scipy.spatial import ConvexHull

from sklearn.gaussian_process import GaussianProcessRegressor

import os
import shutil

from .model import Boundary
from ..utils import (load_pickle, dump_pickle)


class AdaptiveGridSearch(object):
    """Adaptive two-phase grid search. In the initla phase """
    def __init__(self, pars, name, func, widths, adapt_dur, npoints,
                 outfolder=None, nprocs=None):
        self._widths = None
        self._X = None
        self.Xstored = None
        self._Z = None
        self._blobs = None

        self._adapt_dur = adapt_dur
        self._npoints = npoints
        self.func = func
        self.pars = pars
        self.widths = widths
        self.name = name
        self.outfolder = outfolder

        self._initialised = False
        self._counter = 0

    @property
    def X(self):
        return self._X

    @property
    def Z(self):
        return self._Z

    @property
    def blobs(self):
        return self._blobs

    @property
    def widths(self):
        """A dictionary of prior widths for each parameter"""
        return self._widths

    @widths.setter
    def widths(self, widths):
        print(widths)
        print(self.pars)
        try:
            self._widths = {p: Boundary(p, widths[p]) for p in self.pars}
        except KeyError:
            raise ValueError('must provide prior width for each pars')

    @property
    def outfolder(self):
        return self._outfolder

    @outfolder.setter
    def outfolder(self, outfolder):
        if outfolder is None:
            return
        if not outfolder[-1] == '/':
            outfolder += '/'
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        self._outfolder = outfolder

    def _uniform_points(self, Npoints=1, seed=None):
        """Randomly picked points from a uniform distribution over
        the grid
        """
        np.random.seed(seed)
        return np.array([np.random.uniform(self.widths[p].min,
                                           self.widths[p].width, Npoints)
                        for p in self.pars]).T

    def _grid_points(self, nside):
        """Regular grid of points over the prior
        """
        points = np.meshgrid(*[np.linspace(self.widths[p].min,
                               self.widths[p].max, nside)
                               for p in self.pars])
        X = np.vstack([p.reshape(-1,) for p in points]).T
        return X

    def evaluate_function(self, X):
        thetas = list()
        for i in range(X.shape[0]):
            point = {self.pars[j]: X[i, j] for j in range(len(self.pars))}
            thetas.append(point)

        out = [self.func(theta) for theta in thetas]

        Z = [p[0] for p in out]
        blobs = [p[1] for p in out]

        if self._X is None:
            self._X = X
            self._Z = np.array(Z)
            self._blobs = blobs
        else:
            self._X = np.vstack([self._X, X])
            self._Z = np.hstack([self._Z, np.array(Z)])
            for blob in blobs:
                self._blobs.append(blob)
        return Z

    def convex_hull(self):
        Zsorted = np.exp(np.sort(self._Z))
        Zcutoff = Zsorted[np.abs(np.cumsum(Zsorted/np.sum(Zsorted))
                                 - 0.005).argmin()]
        mask = np.where(np.exp(self._Z) > Zcutoff)
        return ConvexHull(self._X[mask, :].reshape(-1, len(self.pars)))

    def _in_hull(self, point, hull, tolerance=1e-12):
        """Returns True if poin in the hull"""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                   for eq in hull.equations)

    def _points_in_hull(self, npoints):
        hull = self.convex_hull()
        newp = list()
        while True:
            p = self._uniform_points().reshape(-1,)
            if self._in_hull(p, hull):
                newp.append(p)

            if len(newp) >= npoints:
                break
        X = np.array(newp)
        return X

    def store_points(self, X):
        self.Xstored = X

    def run(self, nnew=None):
        if not self._initialised:
            # First phase: grid
            nside = int((self._adapt_dur/2)**(1/len(self.pars)))
            X = self._grid_points(nside)
            print('Initial grid search')
            self.evaluate_function(X)
            self._save()
            # First phase: random
            X = self._uniform_points(int(self._adapt_dur/2))
            print('Initial random search')
            self.evaluate_function(X)
            self._save()
            # Second phase: define the hull
            self._hull = self.convex_hull()
            # Sample points within the hull and evaluate
            X = self._points_in_hull(self._npoints)
            print('Initial random search inside the hull')
            self.evaluate_function(X)
            self._save()

            self._initialised = True

        if nnew is not None:
            if not nnew >= 1:
                raise ValueError('nnew must be at least 1')
            # recompute the convex hull
            self._hull = self.convex_hull()
            X = self._points_in_hull(nnew)
            print('Extra random search within the hull')
            self.evaluate_function(X)
            self._save()

    def run_stored(self, npoints=100):
        if self._counter == 0:
            nsampled = 0
        else:
            nsampled = self._counter
        X = self.Xstored[nsampled:nsampled+100]
        self.evaluate_function(X)
        self._counter += npoints
        self._save()

    def _save(self):
        """Save current progress"""
        if self.outfolder is None:
            return
        data = {'X': self.X, 'Z': self.Z, 'blobs': self.blobs}
        fname = '{}/{}_full.p'.format(self.outfolder, self.name)
        if os.path.isfile(fname):
            archive_fname = '{}/{}_backup.p'.format(self.outfolder, self.name)
            shutil.move(fname, archive_fname)
        dump_pickle(fname, data)

    def load(self, fname):
        """Start back from a backup file. Assumes this is post-initialisation
        phase."""
        data = load_pickle(fname)
        self._X = data['X']
        self._Z = data['Z']
        self._blobs = data['blobs']
        self._initialised = True


class GaussianProcess(object):
    def __init__(self, pars, X, Z):
        self.pars = pars
        self.X = X
        self.Z = Z

    def _gp(self):
        """Returns a Gaussian process object for the sampled points"""
        gp = GaussianProcessRegressor(kernel=None)  # None means RBF
        gp.fit(self.X, self.Z.reshape(-1, 1))
        return gp

    def _grid_points(self, nside):
        X = self.X
        points = np.meshgrid(*[np.linspace(np.min(X[:, i]), np.max(X[:, i]),
                                           nside)
                               for i in range(len(self.pars))])
        return np.vstack([p.reshape(-1,) for p in points]).T

    def dist_2D(self, nside=50):
        if not len(self.pars) == 2:
            raise NotImplementedError('Only 2D posteriors supported')
        gp = self._gp()
        X = self._grid_points(nside)
        return X, gp.predict(X).reshape(-1,)

    def marginal_2D(self, par, nside=50):
        """Sets a Gaussian process over the sampled points in 3D, evaluates
        this on a fixed grid and subsequently marginalises over `par`"""
        if not len(self.pars) == 3:
            raise NotImplementedError('Only 3D posteriors supported')
        indxs = [0, 1, 2]
        ind_marg = self.pars.index(par)
        indxs.pop(ind_marg)
        ind1, ind2 = indxs[0], indxs[1]
        # define the grid
        grid = self._grid_points(nside)
        gp = self._gp()
        Z = gp.predict(grid)

        X, Y = np.meshgrid(np.unique(grid[:, ind1]),
                           np.unique(grid[:, ind2]))
        Zmarg = np.zeros_like(X)
        for i in range(nside):
            for j in range(nside):
                mask = np.intersect1d(np.where(grid[:, ind1] == X[i, j]),
                                      np.where(grid[:, ind2] == Y[i, j]))
                Zmarg[i, j] = sum(Z[mask])
        return X, Y, Zmarg
