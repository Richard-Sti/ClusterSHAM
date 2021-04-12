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

"""Parser to read the config file and generate the (paper) model."""

import numpy
import toml
from joblib import load

# Later will delete this import
import sys
sys.path.append('../')

from clustersham.mocks import (AbundanceMatch, Correlator, proxies)
from clustersham.utils import (GaussianClusteringLikelihood, PaperModel)


class PaperModelConfigParser:
    """
    Parses the mpaper model config file, returns initialised
    `clustersham.utils.PaperModel`.

    Parameters
    ----------
        config_path : str
            Path to the toml config file.
    """

    def __init__(self, config_path, sub_id):
        self.cnf = toml.load(config_path)
        self.CF = load(self.cnf['Main']['survey_path'])

    def get_AM(self):
        """
        Initialises `clustersham.mocks.AbundanceMatch` from the config file.
        """
        # Get the mass/luminosity function
        MFpath = self.cnf['AbundanceMatch'].pop('MFpath', None)
        LFpath = self.cnf['AbundanceMatch'].pop('LFpath', None)

        if MFpath is None and LFpath is None:
            raise ValueError("Either 'MFpath' or 'LFpath' must be specified.")
        if MFpath is not None and LFpath is not None:
            raise ValueError("'MFpath' and 'LFpath' given simultaneously.")
        # Loadmeither the LF or MF
        if MFpath is None:
            nd = numpy.load(LFpath)
            faint_end_first = False
            scatter_mult = 2.5
        else:
            nd = numpy.load(MFpath)
            faint_end_first = True
            scatter_mult = 1
        # Get the proxy
        proxy = self.cnf['AbundanceMatch'].pop('proxy', None)
        if proxy is None:
            raise ValueError("'proxy' must be specified in 'AbundanceMatch'")
        try:
            proxy = proxies[proxy]()
        except KeyError:
            raise ValueError("Invalid proxy name '{}'".format(proxy))
        # Make the kwargs object
        kwargs = {'x': nd[:, 0],
                  'phi': nd[:, 1],
                  'faint_end_first': faint_end_first,
                  'scatter_mult': scatter_mult,
                  'halo_proxy': proxy,
                  'boxsize': self.cnf['Main']['boxsize']}
        kwargs.update(self.cnf['AbundanceMatch'])
        return AbundanceMatch(**kwargs)

    def get_correlator(self):
        """
        Initialises `clustersham.mocks.Correlator` from the config file.
        """
        # Get rpbins
        CF = load(self.cnf['Main']['survey_path'])
        kwargs = self.cnf['Correlator']
        kwargs.update({'rpbins': self.CF['rpbins'],
                       'pimax': self.CF['pimax'],
                       'boxsize': self.cnf['Main']['boxsize']})
        return Correlator(**kwargs)

    def get_likelihood(self):
        """
        Initialises `clustersham.utils.GaussianClusteringLikelihood` from
        the config file.
        """
        CF = load(self.cnf['Main']['survey_path'])
        return GaussianClusteringLikelihood(CF['wp'], CF['cov'])

    def __call__(self):
        """
        TO DO:
            - DOCS
        Check that cuts, rpbins and pimax match. Save the cut range in the
        correlation function file ?
        """
        kwargs = {'AM': self.get_AM(),
                  'correlator': self.get_correlator(),
                  'likelihood': self.get_likelihood(),
                  'bounds': self.cnf['Bounds'],
                  'cut_range': self.CF['cut_range'],
                  'Nmocks': self.cnf['Main']['Nmocks'],
                  'seed': self.cnf['Main']['seed']}
        return PaperModel(**kwargs)
