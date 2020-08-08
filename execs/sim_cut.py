from settings import (rpmin, rpmax, nrpbins, pimax,
                      Nmocks, boxsize, subside, npoints, adapt_dur)
import numpy as np
import argparse

import sys
sys.path.append('../')

from PySHAM import mocks
from PySHAM import utils

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--scope", nargs='+', type=float, help='len-2 tuple')
parser.add_argument("--nd_gal", type=str, help='Path to the LF/MF')
parser.add_argument("--nnew", type=int,
                    help='#points after the initial search')
parser.add_argument("--alpha", nargs='+', type=float,
                    help='Range of allowed alpha values')
parser.add_argument("--scatter", nargs='+', type=float,
                    help='Range of allowed scatter values')

args = parser.parse_args()

# process terminal inputs
name = 'NYUmatch'
nd_gal = np.load(args.nd_gal)
scope = args.scope
bounds = {'alpha': (args.alpha[0], args.alpha[1]),
          'scatter': (args.scatter[0], args.scatter[1])}
Nnew = args.nnew
Njobs = 10
# rest of the parameters
pars = ['alpha', 'scatter']
halo_proxy = utils.vvir_proxy
halos = utils.prep_halos(halos_path='../../BAM/hlist_1.00000.npy',
                         tags=['Vmax@Mpeak', 'vvir'])
rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
outfolder = '../../Data/{}/'.format(name)

fname = 'AM_{}_{}'.format(scope[0], scope[1])
wp_fname = '../../Data/{}/ObsCF{}_{}.p'.format(name, scope[0], scope[1])
survey_wp = utils.load_pickle(wp_fname)

# these will get renamed by default.
survey_wp['wp'] = survey_wp['mean_wp']
survey_wp['covmat'] = survey_wp['covmap_wp']
survey_wp.pop('mean_wp')
survey_wp.pop('covmap_wp')

model = mocks.Model(name, pars, scope, halo_proxy, nd_gal, halos, bounds,
                    survey_wp, boxsize, subside, rpbins, pimax, Njobs,
                    Nmocks)

grid_search = mocks.AdaptiveGridSearch(pars, fname, model.logposterior,
                                       bounds, adapt_dur, npoints,
                                       outfolder, Njobs)
# The grid search data gets automatically saved
grid_search.run(Nnew)
# later add support for re-running the cut
