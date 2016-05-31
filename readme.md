# `scsprox`
[![Build Status](https://travis-ci.org/ajfriend/scsprox.svg?branch=master)](https://travis-ci.org/ajfriend/scsprox)

`scsprox` creates fast proximal operators from CVXPY `Problem` objects.

`scsprox` uses CVXPY to form the proximal operator problem and
translate it to the SCS conic input format.
This translation is performed **only once** during the `Prox`
object initialization to save time.
`scsprox` uses CySCS for matrix-factorization-caching and
warm-starting to reduce solve times over many repeated solves,
as occurs, for example, within the alternating-direction method
of multipliers (ADMM) algorithm.

Please also see the [tutorial Jupyter notebook](tutorial.ipynb).

## Installation
Note: currently, may only work with Python 3
- `pip install scsprox`
- optionally, run tests with `py.test --pyargs scsprox`

## Basic Usage
The `Prox` object
```python
from scsprox import Prox
```

creates a fast proximal operator from any
`cvxpy.Problem` and a dictionary whose values are `cvxpy.Variable` objects.

```python
import numpy as np
import cxvpy as cvx

m, n = 200, 100
A = np.random.randn(m,n)
b = np.random.randn(m)
x = cvx.Variable(n)

prob = cvx.Problem(cvx.Minimize(cvx.norm(A*x-b)))

xvars = {'x': x}
prox = Prox(prob, xvars)
```

The `Prox` object comes with a `Prox.do(x0, rho)` method which computes the prox on the input dictionary `x0`, whose keys must match the dictionary that the `Prox` object was created with.

```python
x0 = {'x': np.zeros(n)}
rho = 1.0
x1 = prox.do(x0, rho)
```

## `Prox.info`
`Prox.info` returns a dictionary with status information:
- `info['status']` is the SCS solver status string, usually `Solved` or `Solved/Inaccurate`
- `info['iter']` is the number of SCS iterations performed during the most recent evaluation of `Prox.do`
- `info['setup_time']` is the SCS setup time in seconds, which includes the matrix factorization which is reused across calls to `Prox.do`
- `info['solve_time']` is the SCS solve time in seconds corresponding to
the most recent call to `Prox.do`

## Settings
CySCS settings can be passed as keyword arguments do the `Prox` constructor
or the `Prox.do` method. For example:
- `verbose=True` turns on status information printing during initialization and solves
- `eps=1e-5` changes the SCS solver tolerance to `1e-5`
- `max_iters=400` sets the maximum number of SCS iterations to 400

Changes in settings persist across calls to `Prox.do`.

## Warm-starting
The `Prox` object automatically warm-starts the solve for a call to
`Prox.do` with the solution from the previous call.
This saves time when successive calls are related,
as often happens in ADMM.

You can reset the warm-start variable to `0` (where `0` is the appropriate
vector size for each variable) by calling `Prox.reset_warm_start()`.

## Zero Element
The `Prox` object is aware of the sizes of its prox variables,
and so passing `x0` to `Prox.do` is optional. If omitted,
`x0=None`, or `x0={}`, `Prox` will perform the prox
on the appropriately-sized zero element, which can be seen by
the user by calling `Prox.zero_elem`.

`Prox.zero_elem` will return a `dict` keyed by the variable names, with
either `numpy.array` or `float` (scalar) values.

## `x0` Datatypes
The input `x0` to `Prox.do` must be a dictionary whose values
are either `numpy.array` or `float` objects.

Note that `scsprox` currently only supports 1D `numpy.array` objects.
That is, 2D "matrix" `numpy.arrays` variables are not yet supported.

## Workspace 
The `Prox` object wraps a `cyscs.Workspace` object, which advanced users can access through the `Prox._work` attribute.
