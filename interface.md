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

