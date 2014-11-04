bayesCPD
========

# Preliminaries

BayesCPD is a Python library for performing efficient changepoint detection in
a flexible class of Bayesian models, including conjugate exponential family
models and a number of latent variable models.
Currently, this only supports online changepoint detection.

## Dependencies:
* Python 2.7
* Numpy
* Scipy
* Cython
* Matplotlib
* Pandas
* IPython and IPython Notebook

After cloning the git repository, you'll need to add this directory to your
PYTHONPATH. Then to build, run the script
```bash
$ ./compile.sh
```

# Performing Changepoint Detection

## Overview
To create a changepoint model in general, create a class which inherits from
BaseModel and implement the abstract methods. The main components involved are
implementing the predictive distribution and updating the model's parameters.

There are a few modes of inference currently supported:
* Time-independent inference
* Full changepoint detection
* Efficient changepoint detection approximation

In general, to run inference for a specific model, you create a dictionary
containing the values of the prior parameters with a string naming the variable
and a value.

```python
# Normal-Normal model
from bayesCPD.models.normal import NN

# constant Hazard function
from bayesCPD.utils.Modelutils import constant_hazard

prior_params = {'mu', 0., 'tau', 0.}
```

Then feed those parameters to the model, specifying which mode to use.

```python
model = NN(prior, ivar=1, mode='all')
```

You can either generate data from the prior (or some new set of values)

```python
# generate 1000 datapoints from the prior
model.generate_data(T=1000, hazard=constant_hazard, rate=250)
model.plot_data()
```

![normal-data](https://www.dropbox.com/s/v0awm8n0o8q7gx2/normal-data.png?dl=0)

Or load your own dataset:
```python
model.load_data(my_dataset)
```
Then run inference and plot results:
```python
# run online changepoint detection
model.inference(rate=250)
model.plot_results()
```
![normal-result](https://www.dropbox.com/s/m49v5e6o34ocmj3/normal-result.png?dl=0)

We have implemented several models you can currently run in the models
directory, including univariate and multivariate versions of conjugate
Gaussian models, Gaussian and other mixture models, and latent Dirichet allocation.

See notebooks/normal_example.ipynb for this example for details.
More examples for other models are also in the notebooks directory.

## Implementing a model

(Coming soon)
