from scs_prox.prox_obj import Prox
from scs_prox.examples import example, example2, example3, example_rand

import numpy as np

def test():
    for ex in example, example2, example3:
        prob, x_vars = ex()
        prox = Prox(prob, x_vars, verbose=False)

        x_vals = prox.prox()

def test2():
    m,n = 10, 5
    seed = 0
    prob, x_vars, true_sol = example_rand(m,n,seed)
    prox = Prox(prob, x_vars, verbose=True)

    prox.prox()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] >= 20

    # check that warm-starting worked
    prox.prox()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] == 0

    # reset the warm-start
    prox.reset_warm_start()
    x0 = prox.prox()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] >= 20

    assert isinstance(x0['z'], float)
    assert isinstance(x0['x'], np.ndarray)
    assert isinstance(x0['y'], np.ndarray)

    assert x0['x'].shape == (n,)
    assert x0['y'].shape == (m,)