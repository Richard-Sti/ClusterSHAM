import numpy as np

import argparse
import toml

from PySHAM import surveys


# ---------------------------------------- #
#       Parse the input arguments          #
# ---------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Catalog name", type=str)
parser.add_argument("--nthreads", type=int)
parser.add_argument("--cut", type=float, nargs='+')
parser.add_argument("--Nmult", type=int)

args = parser.parse_args()
config = toml.load('config.toml')

# ---------------------------------------- #
#        Get variables from config         #
# ---------------------------------------- #


rpmin, rpmax, nrpbins, pimax, ncent = [
        config['main'][p] for p in ['rpmin', 'rpmax', 'nrpbins',
                                    'pimax', 'ncent']]
if args.name == 'NYUmatch':
    survey = surveys.NYUSurvey()
    randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
elif args.name == 'NSAmatch':
    survey = surveys.NSASurvey('SERSIC')
    randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
elif args.name == 'NSAmatch_ELPETRO':
    survey = surveys.NSASurvey('ELPETRO')
    randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
else:
    raise ValueError('unsupported survey')

outfolder = './results/{}/'.format(args.name)
out_fname = 'ObsCF{}_{}.p'.format(args.cut[0], args.cut[1])
if args.cut[0] > 0:
    handle = 'logMS'
else:
    handle = 'Mr'

rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
model = surveys.ProjectedCorrelationFunc(survey, randoms_path, outfolder,
                                         args.nthreads, rpbins, pimax, ncent)

# submit
print('Starting from {} to {}'.format(args.cut[0], args.cut[1]))
obs, rands = model.wp_sample(handle, args.cut, args.Nmult)
model.jackknife_wp(obs, rands, args.cut)
