import numpy as np
import cvxpy as cp

from scsprox.scsprox import stuffed_prox, do_prox
from scsprox.examples import example, example2

from scsprox.scs_mapping import form_prox, rand_param_vals


def test():
    prob, x_vars = example()
    problem_data, indmap, solmap = stuffed_prox(prob, x_vars)

    x = 3
    rho = 2     # x should decrease by 1 / rho
    x0_vals = dict(x=x)

    x_vals, scs_info = do_prox(problem_data, indmap, solmap, x0_vals, rho)

    assert np.allclose(x_vals['x'], x - 1.0 / rho, atol=1e-3)


def test2():
    prob, x_vars = example2()

    # These tests as originally implemented were random, meaning
    # only that no explicit seed was set in advance of them. If you
    # want to make them deterministic, just uncomment this seed set.
    # np.random.seed(123)
    for i in range(100):
        compare_proxes(prob, x_vars, i)


def test_matrix():
    np.random.seed(123)
    x = cp.Variable((2, 2))
    prob = cp.Problem(cp.Minimize(cp.norm(x)))
    x_vars = dict(x=x)
    compare_proxes(prob, x_vars, 0)


def compare_proxes(prob, x_vars, i):
    """ Form the prox problem, put in some random data,
    and test that the output from solving with CVXPY/SCS
    is the same as solving with my custom proxer that skips CVXPY.

    Should have identical input data, so output should be *exactly* identical.
    """

    # Let's form the proximal problem.
    problem_data, indmap, solmap = stuffed_prox(prob, x_vars)
    pxprob, x0_vars = form_prox(prob, x_vars)

    # Need to set x0 and rho.
    rand_param_vals(x0_vars)
    rho = 1.0
    x0_vars['__tau'].value = rho / 2

    x0_vals = {}
    for k in x_vars:
        x0_vals[k] = np.atleast_1d(np.squeeze(np.array(x0_vars[k].value)))

    # Solve using straight CVXPY.
    pxprob.solve(solver=cp.SCS, verbose=False)
    cvxsol = {}
    for k, x in x_vars.items():
        cvxsol[k] = np.atleast_1d(np.squeeze(np.array(x.value)))

    # Now solve it using the technique of this package.
    mysol, scs_info = do_prox(problem_data, indmap, solmap, x0_vals, rho)

    # Let's do some comparisons.
    cvx_obj = pxprob.value
    scs_obj = scs_info['pobj']

    SHOW_STDOUT = False
    if SHOW_STDOUT:
        print("\n*********************************************************************************************")
        print("i = {}".format(i))
        print("obj: cvx = {:.6e}  scs = {:.6e}".format(cvx_obj, scs_obj))
    assert np.isclose(cvx_obj, scs_obj, atol=1e-3)

    for k in mysol:
        r = cp.norm(mysol[k] - cvxsol[k]).value
        if SHOW_STDOUT:
            print("{}:\t{}\n\t{}  (r = {:.3e})".format(k, mysol[k], cvxsol[k], r))
        assert np.isclose(r, 0.0, atol=1e-3)
    if SHOW_STDOUT:
        print("*********************************************************************************************\n")

