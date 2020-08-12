import os

import numpy as np

import argparse
import toml

from time import sleep

from PySHAM.mocks import (AdaptiveGridSearch, Model)
from PySHAM import utils

# ---------------------------------------- #
#       Parse the input arguments          #
# ---------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--type", help="AM type", type=str)
parser.add_argument("--cut", type=int)
parser.add_argument("--file_index", type=int, default=0)

args = parser.parse_args()
config = toml.load('config.toml')

# ---------------------------------------- #
#        Get variables from config         #
# ---------------------------------------- #

rpmin, rpmax, nrpbins, boxsize, subside, pimax, Njobs,\
        Nmocks, adapt_dur, npoints = [config['main'][p]
                                      for p in ['rpmin', 'rpmax', 'nrpbins',
                                                'boxsize', 'subside',
                                                'pimax', 'Njobs', 'Nmocks',
                                                'adapt_dur', 'npoints']]
initial = bool(config['main']['initial'])
pars = config[args.name]['pars']
bounds = {p: tuple(config[args.name][args.type][str(args.cut)][p])
          for p in pars}
nd_gal = np.load(config[args.name][args.type]['nd_gal'])
scope = config[args.name][args.type][str(args.cut)]['scope']
proxy = config['main']['proxy']
if proxy == 'vvir':
    proxy = utils.vvir_proxy
    tags = ['Vmax@Mpeak', 'vvir']
elif proxy == 'mvir':
    proxy = utils.mvir_proxy
    tags = ['mvir', 'mpeak']
else:
    raise ValueError('invalid proxy choice')

halos = utils.prep_halos(halos_path=config['main']['halos'], tags=tags)
outfolder = '../Data/{}/'.format(args.name)
out_fname = 'AM{}_{}_{}_{}'.format(args.name, args.file_index,
                                   scope[0], scope[1])
wp_fname = '../Data/{}/ObsCF{}_{}.p'.format(args.name, scope[0], scope[1])

rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
survey_wp = utils.load_pickle(wp_fname)

# these will get renamed by default.
survey_wp['wp'] = survey_wp['mean_wp']
survey_wp['covmat'] = survey_wp['covmap_wp']
survey_wp.pop('mean_wp')
survey_wp.pop('covmap_wp')


# ---------------------------------------- #
#         Initialise the objects           #
# ---------------------------------------- #

model = Model(args.name, pars, scope, proxy, nd_gal, halos, bounds,
              survey_wp, boxsize, subside, rpbins, pimax, Njobs, Nmocks)

grid = AdaptiveGridSearch(pars, out_fname, model.logposterior, bounds,
                          adapt_dur, npoints, outfolder, Njobs)

nside = int((adapt_dur)**(1/len(pars)))
X = grid._grid_points(nside)
grid.store_points(X)

nevals = 100
N = (X.shape[0] // nevals)
# ---------------------------------------- #
#         Start job submissions I          #
# ---------------------------------------- #

fname = './temp/grid_{}'.format(out_fname) + '_{}.p'
comm = ("addqueue -s -q berg -n 1x10 -m 6 /usr/bin/python3 "
        "_simulation.py --fname {}")
# comm = ("python3 _simulation.py --fname {}")
i = 0
fname_sub = fname.format(i)
out = {'grid': grid, 'fname': fname, 'i': i}
utils.dump_pickle(fname_sub, out)
os.system(comm.format(fname_sub))

i += 1
print(comm.format(fname_sub))
while True:
    if i > N:
        break
    if os.path.isfile(fname.format(i)):
        print(i)
        fname_sub = fname.format(i)
        os.system(comm.format(fname_sub))
        print(comm.format(fname_sub))
        i += 1
    sleep(60)

# move the old file
while True:
    if os.path.isfile(fname.format(i)):
        old_fname = './temp/grid_{}'.format(out_fname) + '_{}.p'.format(i)
        new_fname = './temp/grid_{}.p'.format(out_fname)
        os.system('mv {} {}'.format(old_fname, new_fname))
        break
    sleep(60)

# ---------------------------------------- #
#         Start job submissions II         #
# ---------------------------------------- #
print('Going into 2nd phase')

fname = './temp/grid_{}.p'.format(out_fname)
grid = utils.load_pickle(fname)['grid']
grid._counter = 0
X = grid._points_in_hull(npoints)
grid.store_points(X)

N = (X.shape[0] // nevals)

fname = './temp/grid_{}'.format(out_fname) + '_A{}.p'

i = 0
fname_sub = fname.format(i)
out = {'grid': grid, 'fname': fname, 'i': i}
utils.dump_pickle(fname_sub, out)
os.system(comm.format(fname_sub))

i += 1
print(comm.format(fname_sub))
while True:
    if i > N:
        break
    if os.path.isfile(fname.format(i)):
        print(i)
        fname_sub = fname.format(i)
        os.system(comm.format(fname_sub))
        print(comm.format(fname_sub))
        i += 1
    sleep(60)

# move the old file
while True:
    if os.path.isfile(fname.format(i)):
        old_fname = './temp/grid_{}'.format(out_fname) + '_A{}.p'.format(i)
        new_fname = './temp/grid_{}.p'.format(out_fname)
        os.system('mv {} {}'.format(old_fname, new_fname))
        break
    sleep(60)
