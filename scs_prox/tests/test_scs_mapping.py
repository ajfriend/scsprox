import numpy as np

from scs_prox.scs_mapping import get_solmap, extract_sol, form_prox, rand_param_vals
from scs_prox.examples import example, example2, example3

def test_1():

    # todo: test that it works on the boundary...

    prob, x_vars = example()

    pxprob, x0_vars = form_prox(prob, x_vars)

    x = 3
    rho = 2 # x should decrease by 1/rho

    x0_vars['x'].value = x    
    x0_vars['__tau'].value = rho/2.0

    pxprob.solve()

    assert np.allclose(x_vars['x'].value, x-1.0/rho)

def test2():
    ex = example
    prob, x_vars = ex()

    pxprob, x0_vars = form_prox(prob, x_vars)
    rand_param_vals(x0_vars)
    solmap = get_solmap(pxprob, x_vars)