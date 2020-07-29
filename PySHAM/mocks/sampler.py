import numpy as np

from scipy.spatial import ConvexHull

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from .model import Boundary

class AdaptiveGridSearch(object):
    """Adaptive two-phase grid search. In the initla phase """

    def __init__(self, pars, widths, adapt_dur, npoints, func=None):
        self._pars = None
        self._widths = None
        self._X = None
        self._Z = None

        self._adapt_dur = adapt_dur
        self._npoints = npoints
        self.func = func
        self.pars = pars
        self.widths = widths

        self._initialised = False

    @property
    def pars(self):
        """Parameter names"""
        return self._pars

    @pars.setter
    def pars(self, pars):
        if not isinstance(pars, (list, tuple)):
            raise ValueError('must provide a list or tuple of pars')
        if isinstance(pars, tuple):
            pars = list(pars)
        if not all([isinstance(p, str) for p in pars]):
            raise ValueError('all constituent pars must be strs')
        self._pars = pars

    @property
    def widths(self):
        """A dictionary of prior widths for each parameter"""
        return self._widths

    @widths.setter
    def widths(self, widths):
        try:
            self._widths = {p: Boundary(p, widths[p]) for p in self.pars}
        except KeyError:
            raise ValueError('must provide prior width for each pars')

    def _uniform_points(self, Npoints=1, append=True):
        """Randomly picked points from a uniform distribution over
        the grid
        """
        X = np.array([np.random.uniform(self.widths[p].min,
                                        self.widths[p].width, Npoints)
                                        for p in self.pars]).T
        if append:
            if self._X is None:
                self._X = X
            else:
                self._X = np.vstack([self._X, X])
        return X

    def _grid_points(self, nside, append=True):
        """Regular grid of points over the prior
        """
        points = np.meshgrid(*[np.linspace(self.widths[p].min,
                               self.widths[p].max, nside)
                               for p in self.pars])
        X = np.vstack([p.reshape(-1,) for p in points]).T
        if append:
            if self._X is None:
                self._X = X
            else:
                self._X = np.vstack([self._X, X])
        return X

    def evaluate_function(self, X):
        Z = list()
        for i in range(X.shape[0]):
            theta = {self.pars[j]: X[i, j] for j in range(len(self.pars))}
            Z.append(self.func(theta))
        Z = np.array(Z)
        if self._Z is None:
            self._Z = Z
        else:
            self._Z = np.hstack([self._Z, Z])
        return Z


    def convex_hull(self):
        Zcutoff = np.sort(self._Z)[np.abs(np.cumsum(\
                    np.sort(self._Z)/np.sum(self._Z)) - 0.05).argmin()]

        mask = np.where(self._Z > Zcutoff)
        return ConvexHull(self._X[mask, :].reshape(-1, len(self.pars)))

    def _in_hull(self, point, hull, tolerance=1e-12):
        """Returns True if poin in the hull"""
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance)
                   for eq in hull.equations)

    def _points_in_hull(self, npoints):
        newp = list()
        while True:
            p = self._uniform_points(append=False).reshape(-1,)
            if self._in_hull(p, self._hull):
                newp.append(p)

            if len(newp) >= npoints:
                break
        X = np.array(newp)
        self._X = np.vstack([self._X, X])
        return X

    def run(self, nnew=None):
        if not self._initialised:
            # First phase: grid
            X = self._grid_points(int(\
                (self._adapt_dur/2)**(1/len(self.pars))))
            Z =  self.evaluate_function(X)
            # First phase: random
            X = self._uniform_points(int(self._adapt_dur/2))
            Z =  self.evaluate_function(X)
            # Second phase: define the hull
            self._hull = self.convex_hull()
            # Sample points within the hull and evaluate
            X = self._points_in_hull(self._npoints)
            Z = self.evaluate_function(X)

            self._initialised = True

        if not nnew is None:
            if not self._initialised:
                raise ValueError('must run the sampler first')
            if not isinstance(nnew, int):
                raise ValueError('nnew must be integer')
            if not nnew >= 1:
                raise ValueError('nnew must be at least 1')
            X = self._points_in_hull(nnew)
            Z = self.evaluate_function(X)

    def gaussian_process(self):
        gp = GaussianProcessRegressor(kernel=None) # None means RBF
        gp.fit(self._X, self._Z.reshape(-1, 1))
        return gp


