import sys
from joblib import load
import numpy

import toml

sys.path.append('../')

from clustersham.mocks import (AbundanceMatch, Correlator, proxies)
from clustersham.utils import GaussianClusteringLikelihood, PaperModel

class ConfigParser:

    def __init__(self, config_path):
        self.cnf = toml.load(config_path)

    def get_proxy(self):
        proxy = self.cnf['AbundanceMatch'].pop('proxy', None)
        if proxy is None:
            raise ValueError("'proxy' must be specified in 'AbundanceMatch'")
        try:
            return proxies[proxy]()
        except KeyError:
            raise ValueError("Invalid proxy name '{}'".format(proxy))

    def get_boxsize(self):
        # Get the boxsize from main
        try:
            return self.cnf['Main']['boxsize']
        except KeyError:
            raise ValueError("'boxsize' must be specified in 'Main'.")

    def get_AM(self):
        # Get the mass/luminosity function
        MFpath = self.cnf['AbundanceMatch'].pop('MFpath', None)
        LFpath = self.cnf['AbundanceMatch'].pop('LFpath', None)

        if MFpath is None and LFpath is None:
            raise ValueError("Either 'MFpath' or 'LFpath' must be specified "
                             "in 'AbundanceMatch'.")

        if MFpath is None:
            nd = numpy.load(LFpath)
            faint_end_first = False
            scatter_mult = 2.5
        else:
            nd = numpy.load(MFpath)
            faint_end_first = True
            scatter_mult = 1

        kwargs = {'x': nd[:, 0],
                  'phi': nd[:, 1],
                  'faint_end_first': faint_end_first,
                  'scatter_mult': scatter_mult,
                  'halo_proxy': self.get_proxy()}

        kwargs.update({'boxsize': self.get_boxsize()})

        kwargs.update(self.cnf['AbundanceMatch'])
        return AbundanceMatch(**kwargs)

    def get_correlator(self):
        # Get rpbins
        pars = ['rpmin', 'rpmax', 'nrpbins']
        args = [self.cnf['Correlator'].pop(key, None) for key in pars]
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError("'{}' must be specified in 'Correlato''"
                                 .format(pars[i]))
        rpbins = numpy.logspace(numpy.log10(args[0]), numpy.log10(args[1]),
                                args[2] + 1)
        kwargs = self.cnf['Correlator']
        kwargs.update({'rpbins': rpbins})
        kwargs.update({'boxsize': self.get_boxsize()})
        return Correlator(**kwargs)

    def get_likelihood(self):
        """Write a check that makes sure the bins are the same.
        Start saving PIMAX and compare it too.?"""
        try:
            path = self.cnf['Main']['survey_path']
        except KeyError:
            raise ValueError("'survey_path' must be specified in 'Main'.")
        CF = load(path)
        return GaussianClusteringLikelihood(CF['wp'], CF['cov'])

    def __call__(self):
        kwargs = {'AM_generator': self.get_AM(),
                  'likelihood_model': self.get_likelihood(),
                  'cluster_model': self.get_correlator(),
                  'bounds': self.cnf['Bounds'],
                  'cut_range': self.cnf['Main']['cut_range'],
                  'Nmocks': self.cnf['Main']['Nmocks']}
        return PaperModel(**kwargs)




def main():
    parser = ConfigParser('config.toml')
    model = parser()
    print(model)


if __name__ == '__main__':
    main()
