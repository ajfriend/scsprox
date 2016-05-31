import numpy as np

from scsprox.scsprox import stuffed_prox, do_prox
from scsprox.examples import example, example2, example3

from scsprox.scs_mapping import form_prox, rand_param_vals

def test():
    prob, x_vars = example()
    data, indmap, solmap = stuffed_prox(prob, x_vars)

    x = 3
    rho = 2 # x should decrease by 1/rho
    x0_vals = dict(x=x)

    x_vals = do_prox(data, indmap, solmap, x0_vals, rho)

    assert np.allclose(x_vals['x'], x-1.0/rho, atol=1e-3)


def test2():
    ex = example2
    prob, x_vars = ex()

    for i in range(5):
        compare_proxes(prob, x_vars)



def compare_proxes(prob, x_vars):
    """ Form the prox problem, put in some random data,
    and test that the output from solving with CVXPY/SCS
    is the same as solving with my custom proxer that skips CVXPY

    Should have identical input data, so output should be *exactly* identical.
    """
    data, indmap, solmap = stuffed_prox(prob, x_vars)

    pxprob, x0_vars = form_prox(prob, x_vars)

    # need to set x0 and rho
    rho = 1.0
    rand_param_vals(x0_vars)
    x0_vars['__tau'].value = rho/2

    # for k in x0_vars:
    #     print(k, x0_vars[k].value)

    pxprob.solve(solver='SCS')
    cvxsol = {}
    for k in x_vars:
        x = np.atleast_1d(np.squeeze(np.array(x_vars[k].value)))
        cvxsol[k] = x

    # print(cvxsol)

    x0_vals = {}
    for k in x_vars:
        x0_vals[k] = np.atleast_1d(np.squeeze(np.array(x0_vars[k].value)))

    # print(x0_vals)

    mysol = do_prox(data, indmap, solmap, x0_vals, rho)

    # print(mysol)

    for k in mysol:
        assert np.all(mysol[k] == cvxsol[k])

