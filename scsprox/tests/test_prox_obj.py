from scsprox.prox_obj import Prox
from scsprox.examples import example, example2, example3, example_rand

import numpy as np

def test():
    for ex in example, example2, example3:
        prob, x_vars = ex()
        prox = Prox(prob, x_vars, verbose=False)

        x_vals = prox.do()

def test2():
    m,n = 10, 5
    seed = 0
    prob, x_vars, true_sol = example_rand(m,n,seed)
    prox = Prox(prob, x_vars, verbose=False)

    prox.do()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] >= 20

    # check that warm-starting worked
    prox.do()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] == 0

    # reset the warm-start
    prox.reset_warm_start()
    x0 = prox.do()
    assert prox.info['status'] == 'Solved'
    assert prox.info['iter'] >= 20

    # check types and sizes
    assert isinstance(x0['z'], float)
    assert isinstance(x0['x'], np.ndarray)
    assert isinstance(x0['y'], np.ndarray)

    assert x0['x'].shape == (n,)
    assert x0['y'].shape == (m,)

    # check that proximal iteration works
    for k in 'x', 'y', 'z':
        # should not be close to start
        assert not np.allclose(x0[k], true_sol[k], atol=1e-4) 

    prox.work.settings['eps'] = 1e-5
    for _ in range(100):
        x0 = prox.do(x0, verbose=False)

    for k in 'x', 'y', 'z':
        assert np.allclose(x0[k], true_sol[k], atol=1e-4) 
    