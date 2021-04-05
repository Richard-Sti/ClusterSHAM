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
"""Specific posterior for our AM model."""
import numpy as np
from scipy.stats import uniform

import toml

from PySHAM import (mocks, utils, surveys)


class Posterior(object):
    """Posterior for our AM model."""

    def __init__(self, args):
        # parse the arguments
        args_AM = abundance_matching_parser(args)
        args_likelihood = clustering_likelihood_parser(args)
        # add the AM model to the arguments
        AM_model = mocks.AbundanceMatch(**args_AM)
        args_likelihood.update({'AM_model': AM_model})
        # create the likelihood
        self.model_likelihood = mocks.ClusteringLikelihood(**args_likelihood)
        # read the bounds
        self.bounds, self.nside = bounds_parser(args)

    def loglikelihood(self, theta):
        """Log likelihood, assumes just clustering likelihood."""
        return self.model_likelihood.logpdf(theta, return_blobs=True)

    def logprior(self, theta):
        """Assume uniform prior on parameters."""
        lp = 0
        for p in self.model_likelihood.parameters:
            bound = self.bounds[p]
            lp += uniform.logpdf(x=theta[p], loc=bound[0],
                                 scale=abs(bound[0] - bound[1]))
        return lp

    def __call__(self, theta):
        lp = self.logprior(theta)
        if not np.isfinite(lp):
            return lp
        ll, blobs = self.loglikelihood(theta.copy())
        # append the logprior and loglikelihood to the blobs
        blobs.update({'loglikelihood': ll, 'logprior': lp})
        # append the sampled point
        blobs.update({'theta': theta})
        return blobs


def abundance_matching_parser(args):
    """Parses the config gile and returns arguments for
    PySHAM.mocks.AbundanceMatch.
    """
    config = toml.load(args.config)
    main = config['main']
    sampling = config['sampling']

    kwargs = {}
    # get boxsize and Nmocks from main
    for p in ['boxsize', 'Nmocks']:
        kwargs.update(pop_config(main, p))
    # get max scatter
    kwargs.update({'max_scatter': sampling['bounds']['scatter'][1]})
    # get scope. For that we need to know which survey this is
    survey = get_survey(args.name)

    handle = survey.handle(args.nd_type)
    faint_end_first = survey.faint_end_first(handle)
    fraction_bins = pop_config(main, 'fraction_bins', False)

    scopes = survey.scopes(handle=handle, faint_end_first=faint_end_first,
                           fraction_bins=fraction_bins)
    scope = scopes[args.bin_index]
    kwargs.update({'scope': scope})
    # get survey curoffs
    kwargs.update({'survey_cutoffs': survey.cutoffs(handle)})
    # process the halo proxy
    proxy = mocks.proxies[pop_config(sampling, 'proxy', False)]()
    kwargs.update({'halo_proxy': proxy})
    # get nd_gal
    name_nd = pop_config(sampling, 'nd_gal', False).format(args.name,
                                                           args.nd_type)
    kwargs.update({'nd_gal': np.load(name_nd)})
    # get halos
    name_halo = pop_config(sampling, 'halos', False).format(proxy.name)
    kwargs.update({'halos': np.load(name_halo)})
    return kwargs


def clustering_likelihood_parser(args):
    """Parses the config file and returns arguments for
    PySHAM.mocks.ClusteringLikelihood except for ``AM_model``.
    """
    config = toml.load(args.config)
    main = config['main']
    sampling = config['sampling']

    kwargs = {}
    # process main first
    for p in ['pimax', 'subside']:
        kwargs.update(pop_config(main, p))
    # get rpbins
    nrpbins = pop_config(main, 'nrpbins', False)
    rpmin = pop_config(main, 'rpmin', False)
    rpmax = pop_config(main, 'rpmax', False)
    rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
    kwargs.update({'rpbins': rpbins})
    # get survey wp
    fname = pop_config(sampling, 'survey', False).format(
            args.name, args.nd_type, args.bin_index)
    survey = utils.load_pickle(fname)
    kwargs.update({'wp_survey': survey['wp'], 'cov_survey': survey['cov']})
    # get parameters
    kwargs.update(pop_config(sampling, 'parameters'))
    return kwargs


def bounds_parser(args):
    """Parses the config file and looks for parameter boundaries."""
    config = toml.load(args.config)
    bounds = config['sampling']['bounds']
    nside = config['sampling']['nside']
    for p in bounds.keys():
        if len(bounds[p]) != 2:
            raise ValueError("Boundaries for {} must be length 2.".format(p))
        a, b = bounds[p]
        if not b > a:
            bounds[p] = tuple([b, a])
        else:
            bounds[p] = tuple([a, b])
        # Check nside
        nside[p] = int(nside[p])
    return bounds, nside


def pop_config(config, name, return_dict=True):
    """Utility classs to pop items from a dictionary."""
    obj = config.pop(name, None)
    if obj is None:
        raise ValueError("``{}`` must be specified.".format(name))
    if return_dict:
        return {name: obj}
    return obj


def get_survey(name):
    """Returns a survey given some ``name``."""
    if name == 'NYUmatch':
        return surveys.NYU()
    elif name == 'NSAmatch':
        return surveys.NSA('SERSIC')
    elif name == 'NSAmatch_ELPETRO':
        return surveys.NSA('ELPETRO')
    elif name == 'matched':
        return surveys.Matched('ELPETRO')
    else:
        raise ValueError("Invalid sampling name {}.".format(name))
