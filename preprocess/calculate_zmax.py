import numpy
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.optimize import minimize_scalar

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed, parallel_backend

import sys
sys.path.append('../../galofeats/')
sys.path.append('../')
from clustersham.utils import (
        DataSelector, LogRoutine, ApparentMagnitudeRoutine, FiniteCondition,
        RangeCondition, IsEqualCondition)

from galofeats import (UnionPipeline, DataFrameSelector, stratify_split,
                       SklearnRegressor)

njobs = 1


def get_selector():
    survey = fits.open('../data/surveys/nsa_v1.fits')
    survey = survey[1].data

    cosmo = FlatLambdaCDM(H0=100, Om0=0.295)


    routines = {'ELPETRO_LOG_MASS': LogRoutine('ELPETRO_MASS'),
                'ELPETRO_APPMAG': ApparentMagnitudeRoutine(
                    'ZDIST', 'ELPETRO_ABSMAG', cosmo, 'ELPETRO_KCORRECT')}

    conditions = [FiniteCondition('ELPETRO_ABSMAG'),
                  RangeCondition('RA', (108, 270)),
                  IsEqualCondition('IN_DR7_LSS', True),
                  RangeCondition('ELPETRO_APPMAG', ('-oo', 17.6)),
                  RangeCondition('ZDIST', (0, 0.15))]

    indices = {'ELPETRO_ABSMAG': {'column': 4},
               'ELPETRO_KCORRECT': {'column': 4},
               'ELPETRO_MTOL': {'column': 4}}

    selector = DataSelector(survey, conditions, routines, indices)
    return selector, cosmo


def prepare_regressor(selector, model):
    IDS = numpy.where(selector._mask)[0]

    features = ['ZDIST', 'ELPETRO_METS', 'ELPETRO_MTOL', 'ELPETRO_B300']
    target = 'ELPETRO_KCORRECT'
    attrs = features + [target]

    data = numpy.zeros(selector[attrs[0]].size,
                       dtype={'names': attrs, 'formats': [float]*len(attrs)})
    data_features = numpy.zeros(selector[attrs[0]].size,
                                dtype={'names': features,
                                       'formats': [float]*len(features)})

    for attr in attrs:
        data[attr] = selector[attr]
        if attr != 'ELPETRO_KCORRECT':
            data_features[attr] = data[attr]

    pipes = [Pipeline([('selector', DataFrameSelector('ELPETRO_METS')),
                       ('scaler', StandardScaler())]),
            Pipeline([('selector', DataFrameSelector('ZDIST')),
                      ('scaler', StandardScaler())]),
            Pipeline([('selector', DataFrameSelector('ELPETRO_MTOL')),
                      ('scaler', StandardScaler())]),
             Pipeline([('selector', DataFrameSelector('ELPETRO_B300', 'ELPETRO_B300')),
                       ('scaler', StandardScaler())])]
    feature_pipe = UnionPipeline(pipes)

    target_pipe =  UnionPipeline(
            [Pipeline([('selector', DataFrameSelector('ELPETRO_KCORRECT')),
                       ('scaler', StandardScaler())])])

    Xtrain, Xtest, ytrain, ytest = stratify_split(data, features, target,
                                                  False)
    regressor = SklearnRegressor(model, Xtrain, Xtest, ytrain, ytest,
                                 feature_pipe, target_pipe)
    return regressor, data_features


# Get the selector and the selector's cosmology
selector, cosmo = get_selector()
# Initialise the grid/model
model = ExtraTreesRegressor()

param_grid = {'n_estimators': [50, 100, 250, 500],
              'max_depth': [4, 8, None],
              'min_samples_split': [2, 50, 100]}

grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error',
                    n_jobs=njobs, cv=5)
regressor, data_features = prepare_regressor(selector, model)

# Fir the regressor
regressor.fit()
# Print scores and importanes
score = regressor.score()
fimp = regressor.feature_importance(n_repeats=5, n_jobs=njobs)
pimp = regressor.permutation_importance(n_repeats=5, scoring='r2',
                                        n_jobs=njobs)

print('Importances: ')
attrs = regressor.feature_pipeline.attributes
for i, attr in enumerate(attrs):
        print("F: {}: {:.4f} +- {:.4f}".format(attr, fimp[i, 0], fimp[i, 1]))
        print("P: {}: {:.4f} +- {:.4f}".format(attr, pimp[i, 0], pimp[i, 1]))

# Fit zmaxs
M = selector['ELPETRO_ABSMAG']
mlim = 17.6

def zmax_equation(zmax, i):
    """Equation to be minimised to find zmax."""
    # We need to create a strucutred array that will be passed into ExtraTrees
    # scaled, predicted, inverse transformed
    X = numpy.asarray(data_features[i]).reshape(1,)
    X = numpy.copy(X)
    X['ZDIST'] = zmax
    Kcorr = regressor.predict(X)['ELPETRO_KCORRECT'][0]

    lum_dist = cosmo.luminosity_distance(zmax).value
    return abs(M[i] + 25 + 5 * numpy.log10(lum_dist) + Kcorr - mlim)


def minimise_zmax(i):
    """Calls the bounded minimizer."""
    res = minimize_scalar(zmax_equation, bounds=[0.0, 0.15], method='bounded', args=(i,))
    if res['success']:
        return res['x']
    else:
        return -1


N = 10
with Parallel(n_jobs=njobs) as par:
    zmax = par(delayed(minimise_zmax)(i) for i in range(N))
zmax = numpy.array(zmax)

res = numpy.zeros(M.size, dtype={'names': ['IDS', 'zmax'],
                                   'formats': [int, float]})
res['IDS'] = numpy.where(selector._mask)[0]
res['zmax'][:N] = zmax

numpy.save('./zmax_NSA_ELPETRO.npy', res)
