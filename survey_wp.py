import numpy as np

import argparse
import toml

from PySHAM import surveys


# ---------------------------------------- #
#       Parse the input arguments          #
# ---------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--type", help="AM type", type=str)
parser.add_argument("--cut", type=int)

args = parser.parse_args()
config = toml.load('config.toml')

# ---------------------------------------- #
#        Get variables from config         #
# ---------------------------------------- #

scope = config[args.name][args.type][str(args.cut)]['scope']
nmult = config[args.name][args.type][str(args.cut)]['Nmult']
rpmin, rpmax, nrpbins, pimax, Njobs,\
        ncent = [config['main'][p] for p in ['rpmin', 'rpmax', 'nrpbins',
                                             'pimax', 'Njobs', 'ncent']]
if args.name == 'NYUmatch':
    survey = surveys.NYUSurvey()
    randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
elif args.name == 'NSAmatch':
    survey = surveys.NSASurvey('SERSIC')
    randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
else:
    raise ValueError('unsupported survey')

outfolder = './results/{}/'.format(args.name)
out_fname = 'ObsCF{}_{}.p'.format(scope[0], scope[1])
if args.type == 'LF':
    handle = 'Mr'
else:
    handle = 'logMS'

rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
model = surveys.ProjectedCorrelationFunc(survey, randoms_path, outfolder,
                                         Njobs, rpbins, pimax, ncent)

# submit
print('Starting {}'.format(scope))
obs, rands = model.wp_sample(handle, scope, nmult)
model.jackknife_wp(obs, rands, scope)
