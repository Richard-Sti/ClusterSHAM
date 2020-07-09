import numpy as np
import healpy as hp

from .base import BaseSurvey


class NYUSurvey(BaseSurvey):
    def __init__(self):
        self.redshift_range = (0.01, 0.25)
        self.apparent_mr_range = (10.0, 17.7)
        self.survey_area = 7960.0

        self._parse_catalog()

    @property
    def data(self):
        return self._data

    def _parse_catalog(self):
        path = "/mnt/zfsusers/rstiskalek/Data/NYUcatalog_wpols.npy"
        cat = np.load(path)
        try:
            inpolygon = cat['IN_POL']
        except:
            raise ValueError('must provide IN_POL key')
        ngal = cat['RA'][inpolygon].size
        names = ['Mr', 'apMr', 'Kcorr', 'MS', 'RA', 'DEC', 'Z', 'dist']
        data = np.zeros(ngal, dtype={'names':names,\
                'formats':[('float64')]*len(names)})
        try:
            for name in names:
                data[name] = (cat[name])[inpolygon]
        except:
            raise ValueError('some catalog handle is missing')
        masks = list()
        masks.append(np.where(np.logical_and(\
                data['apMr'] > self.apparent_mr_range[0],\
                data['apMr'] < self.apparent_mr_range[1]))[0])
        masks.append(np.where(np.logical_and(\
                data['Z'] > self.redshift_range[0],\
                data['Z'] < self.redshift_range[1]))[0])
        masks.append(np.where(np.isfinite(data['MS']))[0])
        masks.append(np.where(np.isfinite(data['Mr']))[0])
        # Eliminate some outlier galaxies
        theta = np.pi/2-np.deg2rad(data['DEC'])
        phi = np.deg2rad(data['RA'])
        masks.append(np.where(0.15 < hp.rotator.angdist(np.vstack([theta, phi]),\
                [0.16*np.pi, 1.44*np.pi])))
        masks.append(np.where(0.075 < hp.rotator.angdist(np.vstack([theta, phi]),\
                [0.51*np.pi, 1.38*np.pi])))
        masks.append(np.where(0.04 < hp.rotator.angdist(np.vstack([theta, phi]),\
                [np.pi/2, 0.7*np.pi])))
        mask = np.arange(ngal)
        for submask in masks:
            mask = np.intersect1d(mask, submask)

        data = data[mask]
        # Replace MS with logMS
        data['MS'] = np.log10(data['MS'])
        names = list(data.dtype.names)
        i = None
        for j in range(len(names)):
            if names[j] == 'MS':
                i = j
        names[i] = 'logMS'
        data.dtype.names = names
        self._data = data
