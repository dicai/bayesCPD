import numpy as np
import time
import pylab
import pdb

from numpy import random as r
from abc import ABCMeta
from abc import abstractmethod

from bayesCPD.utils.Modelutils import meancollapse
from bayesCPD.utils.Modelutils import modecollapse
from bayesCPD.utils.Modelutils import get_full_row_cython

class BaseModel:

    __metaclass__ = ABCMeta

    def __init__(self, prior_params, cython=False, testing=False, mode='all',
            data=[], verbose='some'):

        self.name = 'Model'
        self.cython = cython
        self.testing = testing
        self.mode = mode

        mtypes = ['single', 'all', 'time_indep']
        if mode not in mtypes:
            raise Exception('Mode must be one of the following:', mtypes)

        self.verbose = verbose
        vtypes = ['none', 'some', 'lots']
        if verbose not in vtypes:
            raise Exception('Verbose must be one of the following:', vtypes)

        # prior and posterior parameters for model
        self.params = Parameters(mode, prior_params)

        # predictive probabilities, predictive means, etc
        self.changept = []
        #self.preds = {}
        #self.preds_reg = {}
        self.preds = []
        self.pred_means = []
        self.pred_vars = []
        self.pred_means_reg = []

        # benchmarking
        self.memory = []
        self.times = []

    @abstractmethod
    def generate_parameters(self, params=None, **kwargs):
        pass

    @abstractmethod
    def get_predprobs(self, datum, **kwargs):
        pass

    @abstractmethod
    def update_parameters(self, datum, **kwargs):
        pass

    @abstractmethod
    def update_others(self, datum, **kwargs):
        pass

    @abstractmethod
    def compute_expecations(self):
        pass

    @abstractmethod
    def plot_preds(self, ax, reg=True, scale=None, dim=-1):
        pass


    def load_data(self, data, cpts=[]):
        self.data = data
        self.cpts = cpts
        try:
            self.D = len(data[0])
        except TypeError:
            self.D = 1

    def generate_data(self, T, hazard, rate, params=None):

        ### if params is None, then generate from the prior
        if params == None:
            prior = self.params.prior_params
        else:
            prior = params

        # initial parameters
        self.generate_parameters(prior)

        ## Generate data without changepoints
        if self.mode == 'time_indep':
            if self.verbose == 'some' or self.verbose == 'all':
                print('Generating data w/o changepoints for %d timesteps.' % T)
            self.data = np.array([self.draw_value() for t in xrange(T)])

        ## Generate data with changepoints
        else:
            if self.mode == 'all':
                if self.verbose == 'some' or self.verbose == 'all':
                    print('Generating data w/ changepoints for %d timesteps.' % T)

                data = []
                changept = []
                # set initial values
                curr_run = 0

                for t in xrange(T):
                    # probability of a changepoint
                    p = hazard(curr_run, rate=rate)
                    if r.random() < p or t == 0:
                        # generate new parameters
                        self.generate_parameters(prior)
                        # reset current run length
                        curr_run = 0
                        # save value of changept
                        changept.append(t)
                    else:
                        curr_run += 1
                    data.append(self.draw_value())

                self.changept = changept

            elif self.mode == 'single':
                # TODO: to be implemented
                pass

            else:
                pass

            self.data = np.array(data)

            if self.verbose == 'some' or self.verbose == 'all':
                print 'Changepoints: ', self.changept

        try:
            self.D = len(self.data[0])
        except TypeError:
            self.D = 1

    #@profile
    def inference(self, B=None, Q=None, rate=250, full=True,
            operation=meancollapse):

        if self.verbose == 'some' or self.verbose == 'all':
            print 'Performing %s inference for %s model' % (self.mode, self.name)

        ### for now let T be length of data
        try:
            if self.data.any() == False:
                    raise Exception('Need to load or generate data')
            else:
                data = self.data
                T = len(self.data)

        # TODO: redo this
        except ValueError:
            if len(self.data.shape) <= 1:
                raise Exception('Need to load or generate data')
            else:
                data = self.data
                T = len(self.data)

        ## Do inference without changepoints
        if self.mode == 'time_indep':

            #predprobs = self.get_predprobs(data[0])
            for t in xrange(0, T):
                timeit1 = time.time()
                self.t = t

                if t % 100 == 0 and self.verbose == 'some':
                    print 'Iteration ', t

                # reindexing data x_1, x_2, ... --> x_0, x_1, ...
                # so data[t] is actually the next obs instead of current
                sstats = self.update_parameters(data[t])
                self.update(sstats)

                if t != T-1:
                    predprobs = self.get_predprobs(data[t+1])
                    self.preds.append(np.log(predprobs.sum()))

                self.params.save_memory()

                timeit2 = time.time()
                self.times.append(timeit2-timeit1)

            self.preds = np.array(self.preds)

            print 'Mean Log Predictive prob', self.preds.mean()

            return

        ## Do inference with changepoints
        else:
            if self.mode == 'all':

                if B == None or Q == None:
                    B = T+1; Q = T+1

                if Q < B - 1:
                    raise Exception("Parameter error: requires Q >= B-1")

                # Initialize p(r_0 = 0) = 1
                R = [np.array([1.])]
                prevR = R[0]

                # tracks number of bins in each queue
                self.qitems = [1]
                self.sitems = [1]
                self.nqueues = len(self.qitems)
                self.t = 0

                # stuff for reconstructing matrix
                Rmat = []

                time1 = time.time()

                #pdb.set_trace()

                # compute predictive distribution
                predprobs = self.get_predprobs(data[0])

                for t in xrange(0, T):
                    ### note on the indexing: the data is 0 indexed, where we
                    ### are letting "prior" values be the index 0, so in the code
                    ### we are going to go "backwards"
                    obs = data[t]
                    self.t = t

                    if full:
                        if Q >= T or Q == None:
                            Rmat.append(np.array(prevR))
                        else:
                            Rmat.append(self.get_full(prevR, B, T))

                    if t % 100 == 0 and self.verbose == 'all':
                        print t

                    timeit1 = time.time()

                    self.qitems[0] += 1; self.sitems[0] += 1

                    H = 1./rate #hazard(range(self.params.length), rate=rate)

                    # p(r_{t+1} | x_{0:t})
                    prevR = self.__compute_run_probs(prevR, predprobs, H)
                    # update with x_t
                    sstats = self.update_parameters(data[t])
                    self.update(sstats)

                    # collapse hypotheses and params
                    self._collapse_sstats(B, Q, t, prevR, operation)
                    prevR = self._collapse_runs(prevR, B, Q, operation)

                    # update any other values, if necessary
                    self.update_others(obs)

                    timeit2 = time.time()

                    self.times.append(timeit2-timeit1)

                    if t != T-1:
                        predprobs = self.get_predprobs(self.data[t+1])
                        # save the log predictive probabilities
                        self.preds.append(np.log((predprobs * prevR).sum()))

                    self.make_pred(prevR)
                    self.params.save_memory()

                    # store run length posterior
                    R.append(prevR)


                # reconstruct row
                if full:
                    if Q >= T or Q == None:
                        Rmat.append(np.array(prevR))
                    else:
                        Rmat.append(self.get_full(prevR, B, T-1))

                time2 = time.time()

                if self.verbose == 'some' or self.verbose == 'all':
                    print 'Finished inference in %.3f seconds' % (time2-time1)

                self.Rmat = np.array(Rmat)
                self.preds = np.array(self.preds)

                if self.verbose == 'some' or self.verbose == 'all':
                    print 'Mean Log Predictive Prob', self.preds.mean()


            elif self.mode == 'single':
                pass
            else:
                pass


    def plot_data(self, ax=None, show=False, scatter=False):
        # most basic plotting of data currently
        T = len(self.data)

        # this case we're generally doing changepoints too
        if ax != None:
            if scatter:
                ax.scatter(xrange(1, T+1), self.data, marker='x', color='gray')
            else:
                ax.plot(xrange(1, T+1), self.data, color='gray')

            if self.pred_means != []:
                self.plot_preds(ax)
                #pylab.plot(xrange(2, len(self.pred_means)+2), self.pred_means)

            # draw changepoint lines
            [pylab.axvline(point, linewidth=1, color='r', linestyle='dotted')
                for point in self.changept]
        else:
            pylab.clf()
            if scatter:
                pylab.scatter(xrange(1, len(self.data)+1), self.data, marker='x')
            else:
                pylab.plot(xrange(1, len(self.data)+1), self.data)

            if self.pred_means != []:
                pylab.plot(xrange(2, len(self.pred_means)+2), self.pred_means)
            for point in self.changept:
                pylab.axvline(point, linewidth=1, color='r', linestyle='dotted')

        pylab.xlim(0, len(self.data))

        if show:
            pylab.show()


    def plot_results(self, R, scatter=False, save=None, figsize=[],
            threshold=True, ymax=500):

        from matplotlib.colors import LogNorm

        pylab.clf()

        fig = pylab.figure(1)
        ax = fig.add_subplot(211)

        t = len(self.data); time = np.arange(len(self.data))

        self.plot_data(ax=ax, show=False, scatter=scatter)

        fig.add_subplot(212, aspect='auto')

        if len(R.shape) == 1:
            im = plot_R(R, len(R), fig=fig, thresh=10e-6)

        else:
            dat = R.T.copy()
            if threshold:
                dat[dat < 10e-6] = 0

            im = pylab.imshow(dat, cmap='gray_r', origin='lower',
                    aspect='auto', norm=LogNorm())

        # draw changepoint lines
        [pylab.axvline(point, linewidth=1, color='r', linestyle='dotted')
            for point in self.changept]

        fig.subplots_adjust(right=0.8)

        # set limits for axes
        pylab.ylabel('Run Length')
        pylab.xlabel('Timesteps')
        pylab.ylim([0,ymax])
        pylab.xlim([0,t+1])

        cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.3])
        fig.colorbar(im, cax=cbar_ax)

        if figsize != []:
            fig.set_size_inches(figsize)

        if save == None:
            pylab.show()
        else:
            pylab.savefig(save)


    def finalize(self):
        pass

    def update(self, sstats):

        if self.mode == 'time_indep':
            for name in sstats.keys():
                self.params.post_params[name] = sstats[name]
        else:
            for name in sstats.keys():
                self.params._add_param(name, sstats[name])
                #self.params.post_params[name] = sstats[name]

    def get_full(self, row, base, T, single=False):

        ### this only works if called in the main inference loop

        if self.mode == 'single':

            finalrow = []
            for k in xrange(self.K):
                inds = np.array(self.qitems).cumsum()
                inds = np.insert(inds, 0, 0)
                newrow = np.array([])

                for i in xrange(len(inds) - 1):
                    start = inds[i]; end = inds[i+1]
                    queue = row[k][start:end]
                    for j in queue:
                        newrow = np.append(newrow, np.repeat(j, base ** i)/
                                float(base**i))

                try:
                    assert newrow.size == self.t + 1
                except AssertionError:
                    pdb.set_trace()

                # make sure this is a valid prob dist
                np.testing.assert_approx_equal(newrow.sum(), 1)

                # pad with zeros until T elements
                finalrow.append(np.append(newrow, np.zeros(T - self.t - 1)))

            return np.array(finalrow)

        else:

            inds = np.array(self.qitems).cumsum()
            inds = np.insert(inds, 0, 0)

            '''
            newrow = np.array([])

            for i in xrange(len(inds) - 1):
                start = inds[i]; end = inds[i+1]
                queue = row[start:end]
                for j in queue:
                    newrow = np.append(newrow, np.repeat(j, base ** i)/
                            float(base**i))
            '''

            newrow = get_full_row_cython(row, inds, base)

            if self.testing:
                assert newrow.size == self.t + 1

                # make sure this is a valid prob dist
                try:
                    np.testing.assert_approx_equal(newrow.sum(), 1)
                except AssertionError:
                    pdb.set_trace()

            # pad with zeros until T elements
            return np.append(newrow, np.zeros(T - self.t))

    @classmethod
    def __compute_run_probs(cls, prevR, predprobs, H):
        rt = np.zeros(len(prevR) + 1)
        preds = predprobs * prevR
        #preds = predprobs[:,0] * prevR
        rt[0] = (H * preds).sum()

        rt[1:] = (1 - H) * preds
        rt /= rt.sum()
        return rt

    def _collapse_sstats(self, base, qsize, t, prevR, operation):

        # iterate thru all queues
        nqueues = len(self.sitems)
        for i in xrange(nqueues):
            if self.sitems[i] > qsize:
                if self.verbose == 'all':
                    print 't:%d , collapsing parameters' % t
                #print 'before length: %d' % len(self.muT)
                self.params.collapse_parameters(i, base, qsize, prevR,
                        operation, self.sitems)
                #print 'after length: %d' % len(self.muT)
                self.sitems[i] -= base
                # count the value in the i+1 queue, create new queue if needed
                try:
                    self.sitems[i+1] += 1
                except:
                    self.sitems.append(1)

    def _collapse_runs(self, probs, base, qsize, operation):
        nqueues = len(self.qitems)
        for i in xrange(nqueues):
            if self.qitems[i] > qsize:
                if self.verbose == 'all':
                    print 't:%d , collapsing run lengths' % self.t
                probs = _merge_first_B(probs, i, base, qsize,
                        meta=self.qitems, operation=operation, args=probs)

                self.qitems[i] -= base
                try:
                    self.qitems[i+1] += 1
                except:
                    self.qitems.append(1)

        return probs

# plots jagged array
def plot_R(R, T, fig, thresh=10e-6):
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=thresh, vmax=1.0)

    for t in xrange(T):
        L = R[t].size
        X = R[t].copy()
        X[R[t] < thresh] = 0
        im = pylab.scatter(t*np.ones(L), np.arange(L), c=X, cmap='gray_r', marker='+', norm=norm)

    pylab.xlim(0, T)
    pylab.ylim(0, T)

    return im

###############################################################################

class Parameters(object):

    def __init__(self, mode, prior_params):

        self.prior_params = {}
        self.post_params = {}
        self.run_probs = []
        self.params_length = 0

        self.__add_init_parameters(prior_params)

        if mode != 'time_indep':
            self.shape_parameters()

    def __add_init_parameters(self, prior_params):
        """
            Takes a dictionary of prior parameters and their initial values
            Initializes posterior parameter values to the prior paramter values

            Input: a dictionary
                key: string
                value: initial value, could be float or vector
        """

        self.prior_params = prior_params.copy()
        self.post_params = {}

        for param in prior_params.keys():
            self.post_params[param] = prior_params[param]

        # if changepoint detection mode, we want to reshape these
        ### TODO ###

    def shape_parameters(self):
        for name in self.post_params.keys():
            param = np.array(self.post_params[name])
            # parameter is an int -> 1D array
            if len(param.shape) == 0:
                self.post_params[name] = np.array([param])
                #self.post_params[name] = np.array([param])[None]
            # parameter is an array -> 1 x ( . ) array
            else:
                self.post_params[name] = param[None]

    def collapse_parameters(self, t, base, qsize, runprobs, operation, sitems):

        for name in self.post_params.keys():
            param = self.post_params[name]
            self.post_params[name] = _merge_first_B(param, t, base, qsize,
                    sitems, operation, args=runprobs)


    def get_full(self):
        pass

    def save_memory(self):
        pass

    def _add_param(self, name, sstat):

        prior = np.array(self.prior_params[name])

        if len(sstat.shape) == 1:
            self.post_params[name] = np.append(prior, sstat)
        else:
            shape = sstat.shape
            T = shape[0]; rest = shape[1:]
            newshape = (T+1,) + rest
            #print newshape

            assert len(shape) == len(newshape)

            self.post_params[name] = np.reshape(np.append(prior, sstat), newshape)


def _merge_first_B(probs, i, base, qsize, meta, operation, args=None):
    """
        merges first B and appends to queue i+1
        creates the queue if i+1 does not exist
    """

    # get the index in probs for queue i
    start = sum(meta[:i])
    # initialize new probs list
    shape = list(probs.shape)
    newprobs = np.zeros(([shape[0]-base+1] + shape[1:]))

    # copy over previous queues
    newprobs[:start] = probs[:start]

    # copy over non-merged values in current queue
    curr_queue = probs[start:start+qsize+1]
    num_left = len(curr_queue[:-base])
    assert num_left == qsize+1-base
    newprobs[start:start+num_left] = curr_queue[:-base]

    # merge "first B" values in current queue and put it in "queue i+1"
    ## assumes that args = prevR (used for sstat collapsing)
    if args != None:
        newprobs[start+num_left] = operation(args[-base:], curr_queue[-base:])
    # used for run length collapsing
    else:
        newprobs[start+num_left] = operation(curr_queue[-base:])
    #newprobs[start+num_left] = operation(probs[start:start+qsize+1][-base:])

    # copy over remaining values
    newprobs[start+num_left+1:] = probs[start+qsize+1:]

    return newprobs




###############################################################################
# General types of models that extend BaseModel
###############################################################################

class ConjugateModel(BaseModel):
    """
    Models with Exponential family likelihoods and conjugate priors.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_parameters():
        pass

    @abstractmethod
    def get_predprobs(self):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    def update_others(self, obs):
        pass

    @abstractmethod
    def compute_expecations():
        pass


class LatentVariableModel(BaseModel):
    """
    Models with latent variables, assumes the use of online VBEM algorithm.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_parameters():
        pass

    @abstractmethod
    def get_predprobs(self):
        self.E_step()

    def update_parameters(self):
        self.M_step()

    @abstractmethod
    def update_others(self):
        self.compute_expectations()
        pass

    @abstractmethod
    def compute_expecations():
        pass

    @abstractmethod
    def E_step():
        pass

    @abstractmethod
    def M_step():
        pass

