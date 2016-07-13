# prox object/function interface

- `prox` should be **callable**; either a function or an object with
a `__call__()` method
- `prox(x0, rho, **settings)` evaluates the prox
- `x0` is a dictionary, keyed by a variable "name", with values
being `numpy.ndarray` objects or `float`s

# optional interface
- `prox.info()` returns a dict
- `prox.zero_elem()` gives the appropirate zero elment (the prox knows the size its operating on)
- `prox.reset_warm_start()` resets the warm-start, if any
- `prox.default_settings()`
- `prox.settings`
- `prox.update_settings(**settings)`

# testing
want:
- `make test` should call the local tests based off local (non installed) files. (this may be different than what i want for Cython)
- upon installing from PyPI, users can call `py.test scsprox` to run all tests,
which will run from installed scsprox and installed tests

## not installed, same dir as `setup.py`
`py.test scsprox -vs` - local
`py.test --pyargs scsprox -vs` - local

## *is* installed, same dir as `setup.py`
`py.test scsprox -vs` - local
`py.test --pyargs scsprox -vs` - local

## is installed, random dir
`py.test scsprox -vs` - not found
`py.test --pyargs scsprox -vs` - installed dir

## not installed, random dir
`py.test scsprox -vs` - not found
`py.test --pyargs scsprox -vs` - not found

locally, have make run `py.test scsprox -vs`
user can run `py.test --pyargs scsprox` to run installed tests

http://pytest.org/latest/goodpractices.html

"Once you are done with your work, you can use tox to make sure that the package is really correct and tests pass in all required configurations."



