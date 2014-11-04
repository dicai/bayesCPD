# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# ipython magic
%matplotlib inline
%load_ext autoreload
%autoreload 2

# <codecell>

# Normal-Normal model
from bayesCPD.models.normal import NN
# constant Hazard function
from bayesCPD.utils.Modelutils import constant_hazard

# <codecell>

# initialize prior hyperparameters
prior = {'mu': 0, 'tau': 1}

# initialize model: Normal likelihood with known precision (ivar), Normal conjugate prior on the mean
model = NN(prior, ivar=1)

# <codecell>

# generate 1000 datapoints from the prior
model.generate_data(T=1000, hazard=constant_hazard, rate=250)
model.plot_data()

# <codecell>

# run online changepoint detection
model.inference(rate=250)

# <codecell>

# plot result
model.plot_results(model.Rmat)

# <codecell>


