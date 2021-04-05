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
"""Projected correlation function for surveys submission script."""
import argparse
import toml

import numpy as np

from posterior import (pop_config, get_survey)

from PySHAM import surveys
from PySHAM.utils import dump_pickle

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--config", help="Config file path", type=str)
parser.add_argument("--nd_type", help="LF or SMF", type=str)
parser.add_argument("--bin_index", help="Index of the bin", type=int)
parser.add_argument("--scope", nargs='+', help="Optional scope of the bin",
                    type=float, default=None)
parser.add_argument("--nthreads", type=int)


def parse_survey(args):
    """ AD"""
    main = toml.load(args.config)['main']

    out = {}
    # get data
    survey = get_survey(args.name)
    handle = survey.handle(args.nd_type)
    faint_end_first = survey.faint_end_first(handle)

    fraction_bins = pop_config(main, 'fraction_bins', False)
    if args.scope is None:
        scopes = survey.scopes(handle=handle, faint_end_first=faint_end_first,
                               fraction_bins=fraction_bins)
        scope = scopes[args.bin_index]
    else:
        scope = args.scope
    print(scope)

    data = survey.scope_selection(scope, handle)
    out.update({'data': data})
    # get randoms
    if args.name == 'matched':
        randoms = np.load("/mnt/zfsusers/rstiskalek/pysham/data/"
                          "RandCatMatched.npy")
        randoms = randoms[randoms['Ang_sep'] < 1.33]
    else:
        randoms = np.load("/mnt/zfsusers/rstiskalek/pysham/data/"
                          "RandCatNYU.npy")
    print('SIZE', randoms.size)
    out.update({'randoms': randoms})
    # get rpbins
    rpmin = pop_config(main, 'rpmin', False)
    rpmax = pop_config(main, 'rpmax', False)
    nrpbins = pop_config(main, 'nrpbins', False)
    rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
    out.update({'rpbins': rpbins})

    for p in ['pimax', 'Njack', 'Nmult']:
        out.update(pop_config(main, p))
    out.update({'nthreads': args.nthreads})
    return out


args = parser.parse_args()
model = surveys.ProjectedCorrelationFunction(**parse_survey(args))

if args.scope is None:
    print('Starting bin {} for {}, {}'.format(args.bin_index, args.name,
                                              args.nd_type))
else:
    print('Starting from {} to {} for {}, {}'.format(
        args.scope[0], args.scope[1], args.name, args.nd_type))

wp, cov = model.wp_jackknife()

# save the results
if args.scope is None:
    fname = ("/mnt/zfsusers/rstiskalek/pysham/results/{}/ObsCF_{}_bin{}.p"
             .format(args.name, args.nd_type, args.bin_index))
else:
    fname = ("/mnt/zfsusers/rstiskalek/pysham/results/{}/ObsCF_{}_{}to{}.p"
             .format(args.name, args.nd_type, *args.scope))
out = {'wp': wp, 'cov': cov}
dump_pickle(fname, out)
