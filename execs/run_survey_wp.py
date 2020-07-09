
import numpy as np
import sys
sys.path.append('../')

from PySHAM import surveys
from PySHAM import mocks

from settings import (rpmin, rpmax, nrpbins, pimax,
                      ncent)

#===========================#
#                           #
#      Configs start        #
#                           #
#===========================#
survey = surveys.NSASurvey('SERSIC')
outfolder = "/mnt/zfsusers/rstiskalek/pysham/data/nsasersic/"
randoms_path = "/mnt/zfsusers/rstiskalek/pysham/data/RandCatNYU.npy"
nthreads = 8
handle = 'logMS'
scopes = [(11.2, 15.0)]
nmults = [50]

#===========================#
#                           #
#      Configs end          #
#                           #
#===========================#

rpbins = np.logspace(np.log10(rpmin), np.log10(rpmax), nrpbins + 1)
model = mocks.ProjectedCorrelationFunc(survey, randoms_path, outfolder,\
        nthreads, rpbins, pimax, ncent)

# Submit jobs
for scope, nmult in zip(scopes, nmults):
    print('Starting {}'.format(scope))
    obs, rands = model.wp_sample(handle, scope, nmult)
    model.jackknife_wp(obs, rands, scope)

