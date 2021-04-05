import numpy as np
import joblib

from AbundanceMatching import (AbundanceFunction, calc_number_densities)

halos = np.load('./data/hlist_1.00000.npy')
#tasks = [{'name': 'NYUmatch', 'nd': 'SMF', 'range': (10.9, 12.5), 'alpha': 0.86, 'scatter': 0.2, 'zcutoff': 10000},
#         {'name': 'NYUmatch', 'nd': 'SMF', 'range': (10.6, 10.9), 'alpha': 1.15, 'scatter': 0.2, 'zcutoff': 10000},
#         {'name': 'NYUmatch', 'nd': 'SMF', 'range': (10.3, 10.6), 'alpha': 1.26, 'scatter': 0.2, 'zcutoff': 10000},
#         {'name': 'NYUmatch', 'nd': 'SMF', 'range': (5, 10.3), 'alpha': 1.45, 'scatter': 0.2, 'zcutoff': 10000}]

tasks = [{'name': 'matched', 'nd': 'BMF', 'range': (5, 12.5), 'alpha': 0.0, 'scatter': 0.42, 'zcutoff': 0.22}]
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (10.6, 11.0), 'alpha': 1.16, 'scatter': 0.27, 'zcutoff': 10000},
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (10.3, 10.6), 'alpha': 1.18, 'scatter': 0.36, 'zcutoff': 10000},
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (5., 10.3), 'alpha': 1.29, 'scatter': 0.61, 'zcutoff': 10000}]



#tasks = [{'name': 'matched', 'nd': 'BMF', 'range': (4, 13), 'alpha': 0.0, 'scatter': 0.42, 'zcutoff': 0.22}]
#         {'name': 'matched', 'nd': 'SMF', 'range': (5, 13), 'alpha': 0.0, 'scatter': 0.43, 'zcutoff': 25},
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (5., 10.3), 'alpha': 1.29, 'scatter': 0.61, 'zcutoff': 10000}]
#         {'name': 'matched', 'nd': 'BMF', 'range': (5.0, 14.0), 'alpha': -6.39, 'scatter': 0.6, 'zcutoff': 10000},
#         {'name': 'matched', 'nd': 'BMF', 'range': (5.0, 14.0), 'alpha': 0.0, 'scatter': 0.42, 'zcutoff': 0.22},
#         {'name': 'matched', 'nd': 'SMF', 'range': (5.0, 14.0), 'alpha': -9.89, 'scatter': 0.83, 'zcutoff': 10000},
#         {'name': 'matched', 'nd': 'SMF', 'range': (5.0, 14.0), 'alpha': 0.0, 'scatter': 0.4, 'zcutoff': 10000},
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (5., 15.0), 'alpha': 1.0, 'scatter': 0.2, 'zcutoff': 10000},]
#         {'name': 'NSAmatch_ELPETRO', 'nd': 'SMF', 'range': (7., 14.0), 'alpha': 1.1, 'scatter': 0.005, 'zcutoff': 10000}]


zmpeak = (1 / halos['mpeak_scale']) - 1

for task in tasks:
    nd = np.load('./results/{}/{}.npy'.format(task['name'], task['nd']))
    m = nd[:, 0] > 0
    af = AbundanceFunction(nd[:,0][m], nd[:,1][m], (3.0, 12.5), faint_end_first=True, faint_end_fit_points=3)
    scatter = task['scatter']
    alpha = task['alpha']
    zcut = task['zcutoff']
    preselection = zmpeak < zcut

    remainder = af.deconvolute(scatter, 20)
    plist = halos['mvir'] * (halos['mpeak'] / halos['mvir'])**alpha
    plist = plist[preselection]
    nd_halos = calc_number_densities(plist, 400)

    # do abundance matching with no scatter
    catalog = af.match(nd_halos)

    # do abundance matching with some scatter
    catalog = af.match(nd_halos, scatter)
    x0 = task['range'][0]
    xf = task['range'][1]
    mask = (~np.isnan(catalog)) & (x0 < catalog) & (catalog < xf)
#    mask = (~np.isnan(catalog))
    task.update({'catalog': catalog[mask],
                 'mvir': np.log10(halos['mvir'][preselection][mask]),
                 'mpeak': np.log10(halos['mpeak'][preselection][mask])})

# for task in tasks:
#     nd = np.load('./results/{}/{}.npy'.format(task['name'], task['nd']))
#     af = AbundanceFunction(nd[:,0], nd[:,1], (6, 14.0), faint_end_first=True)
#     alpha = task['alpha']
#     scatter = task['scatter']
# 
# 
#     remainder = af.deconvolute(scatter, 20)
#     plist = halos['mvir'] * (halos['mpeak'] / halos['mvir'])**alpha
#     nd_halos = calc_number_densities(plist, 400)
# 
#     # do abundance matching with no scatter
#     catalog = af.match(nd_halos)
# 
#     # do abundance matching with some scatter
#     catalog = af.match(nd_halos, scatter)
#     x0 = task['range'][0]
#     xf = task['range'][1]
# #    mask = (~np.isnan(catalog)) & (x0 < catalog) & (catalog < xf)
#     mask = (~np.isnan(catalog))
#     task.update({'catalog': catalog[mask],
#                  'mvir': np.log10(halos['mvir'][mask]),
#                  'mpeak': np.log10(halos['mpeak'][mask])})


joblib.dump(tasks, '_halo2galaxymass.z')
