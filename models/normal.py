from scipy.stats import norm
from scipy.stats import t
from numpy import random as r
import numpy as np
import pylab

from bayesCPD.models.basic import ConjugateModel


"""
Contains several models:
    - Normal model with known precision (Normal prior)
    - Normal model with unknown mean and precision (Gamma prior)
    - Multivariate normal with unknown mean and precision (Wishart prior)
"""

# Normal model with known precision
class NN(ConjugateModel):

    def __init__(self, prior_params, ivar, mode='all', seed=101, verbose='some',
            cython=True):

        super(self.__class__, self).__init__(prior_params, mode=mode,
                verbose=verbose, cython=cython)

        np.random.seed(seed)
        self.name = 'Normal Conjugate Model, Known Precision'

        # known value
        self.ivar = ivar

        # for data generation
        self.curr_mean = 0
        self.all_means = []

    def generate_parameters(self, params):

        mu0 = params['mu']
        tau0 = params['tau']

        # draw mean
        self.curr_mean = r.normal(mu0, tau0**-.5)
        self.all_means.append(self.curr_mean)

    def draw_value(self):
        return r.normal(self.curr_mean, self.ivar**-.5)

    def get_predprobs(self, datum):
        """
        Predictive distribution of NN is a Normal.
        """
        muT = self.params.post_params['mu']
        tauT = self.params.post_params['tau']
        return norm.pdf(datum, muT, (1./tauT + 1./self.ivar)**.5)

    def make_pred(self, runprobs):

        muT = self.params.post_params['mu']
        #tauT = self.params.post_params['tau']

        self.pred_means.append((muT * runprobs).sum())
        #self.pred_vars.append(((1./(tauT)+1./(self.ivar)) * runprobs).sum())

        self.pred_means_reg.append(muT[-1])
        #self.pred_vars_reg.append(1./tauT[-1] + 1./self.ivar)

    def plot_preds(self, ax, scale=1):

        T = len(self.pred_means)
        pylab.plot(xrange(1, T+1), self.pred_means, color='blue', linewidth=1.5)
        pylab.plot(xrange(1, T+1), self.pred_means_reg, color='orange', linewidth=2,
                linestyle='dashed')


    def update_parameters(self, datum):

        muT = self.params.post_params['mu']
        tauT = self.params.post_params['tau']

        sstats = {'mu': (tauT * muT + self.ivar * datum) / (tauT + self.ivar),
                  'tau': tauT + self.ivar
                 }

        return sstats

    def compute_expecations(self):
        return {'mu': self.params.post_params['mu']}

# univariate version
class NG(ConjugateModel):

    def __init__(self, prior_params, mode='all', seed=101, verbose='some',
            cython=True):

        super(self.__class__, self).__init__(prior_params, mode=mode,
                verbose=verbose, cython=cython)

        np.random.seed(seed)
        self.name = 'Normal Gamma'

        # for data generation
        self.curr_mean = 0
        self.all_means = []
        self.all_ivars = []

    def generate_parameters(self, params):

        mu0 = params['mu']
        nu0 = params['nu']
        alpha0 = params['alpha']
        beta0 = params['beta']

        # draw precision
        ivar = r.gamma(alpha0, beta0)
        self.curr_ivar = ivar

        # draw mean
        self.curr_mean = r.normal(mu0, (nu0*ivar)**-.5)
        self.all_means.append(self.curr_mean)
        self.all_ivars.append(self.curr_ivar)

    def draw_value(self):
        return r.normal(self.curr_mean, self.curr_ivar**-.5)

    def get_predprobs(self, datum):
        """
        Predictive distribution of NIG is a T distribution.
        """
        muT = self.params.post_params['mu']
        nuT = self.params.post_params['nu']
        alphaT = self.params.post_params['alpha']
        betaT = self.params.post_params['beta']
        return t.pdf(datum, 2*alphaT, muT, betaT*(1+nuT) / (alphaT*nuT))

    def make_pred(self, runprobs):

        muT = self.params.post_params['mu']
        self.pred_means.append((muT * runprobs).sum())
        self.pred_means_reg.append(muT[-1])

    def plot_preds(self, ax, scale=1):

        T = len(self.pred_means)
        pylab.plot(xrange(1, T+1), self.pred_means, color='blue', linewidth=1.5)
        pylab.plot(xrange(1, T+1), self.pred_means_reg, color='orange', linewidth=2,
                linestyle='dashed')

    def update_parameters(self, datum):

        muT = self.params.post_params['mu']
        nuT = self.params.post_params['nu']
        alphaT = self.params.post_params['alpha']
        betaT = self.params.post_params['beta']

        sstats = {'mu': (nuT*muT + datum) / (nuT + 1),
                  'nu': nuT + 1,
                  'alpha': alphaT + 1./2,
                  'beta': betaT + 1.*nuT/(nuT+1) * 1.*(datum-muT)**2 / 2.
                 }

        return sstats

    def compute_expecations(self):

        alpha = self.params.post_params['alpha']
        beta = self.params.post_params['beta']

        return {'E_mu': self.params.post_params['mu'],
                'E_tau': 1.*alpha / (alpha+beta)}


