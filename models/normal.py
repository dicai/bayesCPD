from scipy.stats import norm
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

        super(ConjugateModel, self).__init__(prior_params, mode=mode,
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
        #return np.exp(norm.logpdf(datum, muT, (1./tauT + 1./self.ivar)**.5))

    def make_pred(self, runprobs):

        muT = self.params.post_params['mu']
        tauT = self.params.post_params['tau']

        self.pred_means.append((muT * runprobs).sum())
        self.pred_vars.append(((1./(tauT)+1./(self.ivar)) * runprobs).sum())

        self.pred_means_reg.append(muT[-1])
        #self.pred_vars_reg.append(1./tauT[-1] + 1./self.ivar)

    def plot_preds(self, ax, scale=1):

        T = len(self.pred_means)
        pylab.plot(xrange(2, T+2), self.pred_means, color='blue', linewidth=1.5)

        #pred_vars = np.array(self.pred_vars)
        scales = np.array(self.pred_vars) ** 0.5
        #T = pred_means.size + 1

        pylab.plot(xrange(2, T+2), self.pred_means_reg, color='orange', linewidth=2,
                linestyle='dashed')


    def update_parameters(self, datum):

        #mu0 = self.params.prior_params['mu']
        #tau0 = self.params.prior_params['tau']

        muT = self.params.post_params['mu']
        tauT = self.params.post_params['tau']

        sstats = {'mu': (tauT * muT + self.ivar * datum) / (tauT + self.ivar),
                  'tau': tauT + self.ivar
                 }

        return sstats

    def compute_expecations(self):
        pass
