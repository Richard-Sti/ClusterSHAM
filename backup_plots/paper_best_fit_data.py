"""Makes data for paper illustration plots."""
import numpy as np

from PySHAM import (mocks, utils, surveys)

nthreads = 10
rpbins = np.logspace(np.log10(0.1), np.log10(25), 12 + 1)
halos = np.load('./data/halos_mvir_proxy.npy')
subside = 25
pimax = 60
boxsize = 400
max_scatter = 1.0
Nmocks = 50
cutoffs = (-27.5, -15.0)
parameters = ['alpha', 'scatter']
proxy = mocks.VirialMassProxy()


names = ['NYUmatch', 'NSAmatch_ELPETRO', 'NSAmatch']
nd_types = ['SMF', 'LF']

pars = {'NYUmatch_LF': {'alpha': 1.12, 'scatter': 0.22},
        'NYUmatch_SMF': {'alpha': 1.26, 'scatter': 0.24},
        'NSAmatch_LF': {'alpha': 1.15, 'scatter': 0.28},
        'NSAmatch_SMF': {'alpha': 1.22, 'scatter': 0.31},
        'NSAmatch_ELPETRO_LF': {'alpha': 1.06, 'scatter': 0.24},
        'NSAmatch_ELPETRO_SMF': {'alpha': 1.19, 'scatter': 0.27},
        }


def get_model(name, nd_type, bin_index):
    nd_gal = np.load('./results/{}/{}.npy'.format(name, nd_type))
    # The survey here is arbitrary, we're not interested in the likelihood
    survey = utils.load_pickle('./results/{}/ObsCF_{}_bin{}.p'.format(
        name, nd_type, bin_index))
    wp_survey = survey['wp']
    cov_survey = survey['cov']
    # Get the scope
    if name == 'NYUmatch':
        cat = surveys.NYU()
    elif name == 'NSAmatch':
        cat = surveys.NSA('SERSIC')
    elif name == 'NSAmatch_ELPETRO':
        cat = surveys.NSA('ELPETRO')
    fraction_bins = [0.0, 0.015, 0.15, 0.45, 0.9]
    if nd_type == 'LF':
        scopes = cat.scopes('Mr', False, fraction_bins)
    elif nd_type == 'SMF':
        scopes = cat.scopes('logMS', True, fraction_bins)

    scope = scopes[bin_index]
    scope = (min(scope), max(scope))
    print('Scope is ', scope)
    AM_model = mocks.AbundanceMatch(nd_gal=nd_gal, scope=scope, halos=halos,
                                    boxsize=boxsize, halo_proxy=proxy,
                                    max_scatter=max_scatter, Nmocks=Nmocks,
                                    survey_cutoffs=cutoffs, nthreads=nthreads)

    model = mocks.ClusteringLikelihood(parameters, wp_survey, cov_survey,
                                       AM_model, subside, rpbins, pimax,
                                       nthreads=nthreads)
    return model


out = {}
for name in names:
    for nd_type in nd_types:
        for bin_index in range(4):
            model = get_model(name, nd_type, bin_index)
            theta = pars['{}_{}'.format(name, nd_type)]
            ll, blobs = model.logpdf(theta.copy())
            blobs.update({'theta': theta, 'll': ll, 'name': name,
                          'nd_type': nd_type})
            out['{}_{}_{}'.format(name, nd_type, bin_index)] = blobs

utils.dump_pickle('./results/paper_best_fit_points.p', out)
