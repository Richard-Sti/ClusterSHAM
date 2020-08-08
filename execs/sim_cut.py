import numpy as np

import argparse
import toml

import sys
sys.path.append('../')

from PySHAM import mocks
from PySHAM import utils


parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--type", help="AM type", type=str)
parser.add_argument("--cut", type=int)
parser.add_argument("--file_index", type=int)
parser.add_argument("--phase", type=int)

args = parser.parse_args()
config = toml.load('config.toml')
pars = config['main']


nd_gal = np.load(config[args.name][args.type]['nd_gal'])
scope = config[args.name][args.type][str(args.cut)]['scope']

bounds = {'alpha': config[args.name][args.type][str(args.cut)]['alpha'],
          'scatter': config[args.name][args.type][str(args.cut)]['scatter']}


pars = list(bounds.keys())

if pars['proxy'] == 'vvir':
    halo_proxy = utils.vvir_proxy
    tags = ['Vmax@Mpeak', 'vvir']
elif pars['proxy'] == 'mvir':
    halo_proxy = utils.mvir_proxy
    tags = ['mvir', 'mpeak']
else:
    raise ValueError('invalid proxy choice')


halos = utils.prep_halos(halos_path=pars['halos_path'],
                         tags=tags)
rpbins = np.logspace(np.log10(pars['rpmin']), np.log10(pars['rpmmax']),
                     pars['nrpbins'] + 1)

outfolder = '../../Data/{}/'.format(args.name)

fname = 'AM{}_{}_{}'.format(args.file_index, scope[0], scope[1])

wp_fname = '../../Data/{}/ObsCF{}_{}.p'.format(args.name, scope[0], scope[1])
survey_wp = utils.load_pickle(wp_fname)

# these will get renamed by default.
survey_wp['wp'] = survey_wp['mean_wp']
survey_wp['covmat'] = survey_wp['covmap_wp']
survey_wp.pop('mean_wp')
survey_wp.pop('covmap_wp')

model = mocks.Model(args.name, pars, scope, halo_proxy, nd_gal, halos, bounds,
                    survey_wp, pars['boxsize'], pars['subside'], rpbins,
                    pars['pimax'], pars['Njobs'], pars['Nmocks'])

grid_search = mocks.AdaptiveGridSearch(pars, fname, model.logposterior,
                                       bounds, pars['adapt_dur'],
                                       pars['npoints'], outfolder,
                                       pars['Njobs'])
# The grid search data gets automatically saved
if args.phase != 0:

grid_search.run_phase(args.phase, main['adapt_dur'],
        initial=main['initial'])
#grid_search.run(Nnew)

# decide what points you want to sample ..


# later add support for re-running the cut


















