"""Makes data for paper illustration plots."""
import numpy as np

from PySHAM import mocks, utils


rpbins = np.logspace(np.log10(0.1), np.log10(25), 12 + 1)
halos = np.load('./data/halos_zmpeak_mvir_proxy.npy')
scope = (-22.0, -21.0)
subside = 25
pimax = 60
boxsize = 400
max_scatter = 0.5
Nmocks = 50
cutoffs = (-27.5, -15.0)
parameters = ['alpha', 'scatter', 'zcutoff']

proxy = mocks.ZmpeakVirialMassProxy()


# vary just AM parameters, not the number density
nd_gal = np.load('./results/NYUmatch/LF.npy')
survey = utils.load_pickle('./results/NYUmatch/backup/ObsCF-22.0to-21.0.p')
wp_survey = survey['wp']
cov_survey = survey['cov']

AM_model = mocks.AbundanceMatch(nd_gal=nd_gal, scope=scope, halos=halos,
                                boxsize=boxsize, halo_proxy=proxy,
                                max_scatter=max_scatter, Nmocks=Nmocks,
                                survey_cutoffs=cutoffs, nthreads=10)

model = mocks.ClusteringLikelihood(parameters, wp_survey, cov_survey, AM_model,
                                   subside, rpbins, pimax, nthreads=10)
thetas = list()
# alphas
for i in [1.0, 0.75, 1.25]:
    thetas.append({'alpha': i, 'scatter': 0.005, 'zcutoff': np.infty})
# scatters
for i in [0.005, 0.2, 0.5]:
    thetas.append({'alpha': 1.0, 'scatter': i, 'zcutoff': np.infty})
# zcutoffs
for i in [np.infty, 1.0, 0.25]:
    thetas.append({'alpha': 1.0, 'scatter': 0.005, 'zcutoff': i})

out = [None] * len(thetas)
for i, theta in enumerate(thetas):
    ll, blobs = model.logpdf(theta.copy())
    blobs.update({'theta': theta, 'll': ll})
    out[i] = blobs


utils.dump_pickle('./results/paper_illustration_vary_params.p', out)

## vary input LFs
#out = [None] * 3
#for i, name in enumerate(['NYUmatch', 'NSAmatch', 'NSAmatch_ELPETRO']):
#    nd_gal = np.load('./results/{}/LF.npy'.format(name))
#    survey = utils.load_pickle('./results/{}/backup/ObsCF-22.0to-21.0.p'
#                               .format(name))
#    wp_survey = survey['wp']
#    cov_survey = survey['cov']
#
#    AM_model = mocks.AbundanceMatch(nd_gal=nd_gal, scope=scope, halos=halos,
#                                    boxsize=boxsize, halo_proxy=proxy,
#                                    max_scatter=max_scatter, Nmocks=Nmocks,
#                                    survey_cutoffs=cutoffs, nthreads=10)
#    model = mocks.ClusteringLikelihood(parameters, wp_survey, cov_survey,
#                                       AM_model, subside, rpbins, pimax,
#                                       nthreads=10)
#
#    theta = {'alpha': 1.0, 'scatter': 0.005, 'zcutoff': np.infty}
#    ll, blobs = model.logpdf(theta.copy())
#    blobs.update({'theta': theta, 'll': ll, 'name': name})
#    out[i] = blobs
#
#utils.dump_pickle('./results/paper_illustration_vary_LF.p', out)
