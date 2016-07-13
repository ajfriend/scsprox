from scsprox import Prox
from scsprox.examples import example

import pytest

def test1():
    prob, xvars = example()
    prox = Prox(prob, xvars)

    assert prox.settings == dict(eps=1e-3, max_iters=100, verbose=False)

    prox.update_settings(eps=1e-7)
    assert prox.settings['eps'] == 1e-7

    with pytest.raises(ValueError):
        prox.update_settings(cows=17)

    # won't catch the bad key here
    prox.settings['cows'] = 17
    # but will catch it when prox is run
    with pytest.raises(ValueError):
        prox()

def test_defaults():
    prob, xvars = example()
    prox = Prox(prob, xvars)

    assert prox.settings == dict(eps=1e-3, max_iters=100, verbose=False)

    # make sure the CySCS settings are set to the correct defaults
    assert prox._work.settings['eps'] == 1e-3
    assert prox._work.settings['max_iters'] == 100
    assert prox._work.settings['verbose'] == False



