#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
from scipy.constants import speed_of_light
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

from kmeans_radec import kmeans_sample
from time import time

from PySHAM.utils import dump_pickle

class ProjectedCorrelationFunc(object):
    def __init__(self, survey, randoms_path, outfolder, nthreads, rpbins,\
            pimax=60.0, ncent=300):
        self.survey = survey
        self.outfolder = outfolder
        self.randoms_path = randoms_path
        self.nthreads = nthreads
        self.rpbins = rpbins
        self.nbins = len(rpbins) - 1

        self.ncent = ncent
        self.pimax = pimax

    def wp_sample(self, par, scope, nmult=50):
        """Returns random points and galaxies corresponding to given cut
        """
        # Get observed gals
        mask = np.where(np.logical_and(\
                self.survey.data[par] > scope[0],\
                self.survey.data[par] < scope[1]))
        obs = {'RA' : self.survey.data['RA'][mask],
               'DEC': self.survey.data['DEC'][mask],
               'CZ' : self.survey.data['Z'][mask] * speed_of_light * 1e-3,}
        ngal = mask[0].size
        randsky = np.load(self.randoms_path)
        rands = {'RA' : randsky['RA'][:nmult*ngal],
                 'DEC': randsky['DEC'][:nmult*ngal],
                 'CZ' : np.random.choice(obs['CZ'], nmult*ngal, replace=True),}
        ncluster = mask[0].size
        mask = np.random.choice(np.arange(ncluster), ncluster, False)
        X = np.vstack([rands['RA'][mask], rands['DEC'][mask]]).T
        kmeans = kmeans_sample(X, self.ncent, maxiter=250, tol=1.0e-5,\
                            verbose=0)

        obs['labels'] = kmeans.find_nearest(np.vstack([obs['RA'],\
                                            obs['DEC']]).T)
        rands['labels'] = kmeans.find_nearest(np.vstack([rands['RA'],
                                            rands['DEC']]).T)

        obs['weights'] = np.ones_like(obs['RA'])
        rands['weights'] = np.ones_like(rands['RA'])

        return obs, rands

    def leave_one_out_wp(self, obs, rands, index_leftout):
        mask_obs = np.where(obs['labels'] != index_leftout)
        mask_rand = np.where(rands['labels'] != index_leftout)

        obsra = obs['RA'][mask_obs]
        obsdec = obs['DEC'][mask_obs]
        obscz = obs['CZ'][mask_obs]
        obsweights = obs['weights'][mask_obs]
        nobs = obsra.size

        randra = rands['RA'][mask_rand]
        randdec = rands['DEC'][mask_rand]
        randcz = rands['CZ'][mask_rand]
        randweights = rands['weights'][mask_rand]
        nrands = randra.size

        cosmology = 2 #planck cosmology
        # Auto pair counts in DD i.e. survey catalog
        autocorr = 1
        dd_counts = DDrppi_mocks(autocorr, cosmology, self.nthreads,\
                        self.pimax, self.rpbins, obsra, obsdec, obscz,\
                        obsweights, weight_type='pair_product')
        # Cross pair counts in DR
        autocorr = 0
        dr_counts = DDrppi_mocks(autocorr, cosmology, self.nthreads,\
                        self.pimax, self.rpbins, obsra, obsdec, obscz,\
                        RA2=randra, DEC2=randdec, CZ2=randcz,\
                        weights1=obsweights, weights2=randweights,\
                        weight_type='pair_product')
        # Auto pairs counts in RR i.e. random catalog
        autocorr = 1
        rr_counts = DDrppi_mocks(autocorr, cosmology, self.nthreads,\
                        self.pimax, self.rpbins, randra, randdec, randcz,\
                        randweights, weight_type='pair_product')
        # All the pair counts are done, get the project correlation function
        return convert_rp_pi_counts_to_wp(nobs, nobs, nrands, nrands,\
                dd_counts, dr_counts, dr_counts,\
                rr_counts, self.nbins, self.pimax)

    def jackknife_wp(self, obs, rands, scope):
        wps = list()
        extime = list()
        for i in range(self.ncent):
            start = time()
            wps.append(self.leave_one_out_wp(obs, rands, i))
            extime.append(time() - start)
            remtime = sum(extime)/len(extime)*(self.ncent - i - 1) / 60**2
            print("Done with {}/{}. Estimated remaining time is "
                  "{:.2f} hours".format(1 + i, self.ncent, remtime))
            sys.stdout.flush()
        wps = np.array(wps)
        wp_mean = np.mean(wps, axis=0)
        # bias=True means normalisation by 1/N
        cov = np.cov(wps, rowvar=False, bias=True)*(self.ncent-1)
        self.save_data(wps, wp_mean, cov, scope)
        self.diagnostic_plots(obs, rands, scope)
        return wp_mean, cov

    def save_data(self, wps, wp, cov, scope):
        data = {'wp' : wp, 'cov' : cov, 'wps' : wps, 'cbins' :\
        [0.5*(self.rpbins[i+1] + self.rpbins[i]) for i in range(self.nbins)]}
        dump_pickle(self.outfolder + 'ObsCF{}_{}to{}.p'.format(\
                self.survey.name, scope[0], scope[1]), data)

    def diagnostic_plots(self, obs, rands, scope):
        plt.figure(dpi=240, figsize=(12, 8))
        plt.subplot(221)
        plt.title('obs')
        for label in np.unique(obs['labels']):
            mask = np.where(obs['labels'] == label)
            plt.scatter(obs['RA'][mask], obs['DEC'][mask], s=0.1)

        plt.subplot(222)
        plt.title('rands')
        for label in np.unique(rands['labels']):
            mask = np.where(rands['labels'] == label)
            plt.scatter(rands['RA'][mask], rands['DEC'][mask], s=0.1)

        plt.subplot(223)
        bins = np.linspace(obs['CZ'].min(), obs['CZ'].max(), 50)/3e5
        plt.hist(obs['CZ']/3e5, bins=bins, histtype='step', label='obs',
                 density=1)
        plt.hist(rands['CZ']/3e5, bins=bins, histtype='step', label='rands',
                 density=1)
        plt.legend()
        plt.savefig(self.outfolder + 'plots/CFcheck{}_{}to{}.png'.format(\
                self.survey.name, scope[0], scope[1]))
        plt.close()
