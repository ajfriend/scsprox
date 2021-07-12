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

    x_vals = do_prox(problem_data, indmap, solmap, x0_vals, rho)

    assert np.allclose(x_vals['x'], x - 1.0 / rho, atol=1e-3)


# def test2():
#     prob, x_vars = example2()
#     cvxsol = {}
#     for i in range(100):
#         pxprob, x0_vars = form_prox(prob, x_vars)
#
#         # need to set x0 and rho
#         rho = 1.0
#         x0_vars['x'].value = np.array([2, 3])
#         x0_vars['y'].value = np.array([4, 5])
#         x0_vars['__tau'].value = rho / 2
#
#         pxprob.solve(solver=cp.SCS, verbose=True)
#         cvxsol[i] = {}
#         for k in x_vars:
#             x = np.atleast_1d(np.squeeze(np.array(x_vars[k].value)))
#             cvxsol[i][k] = x
#
#         if i > 0:
#             # Check this run and previous one.
#             print("\ni = {}".format(i))
#             for k in x_vars:
#                 r = cp.norm(cvxsol[i][k] - cvxsol[i - 1][k]).value
#                 print("{}:\t{}\n\t{}  (r = {:.3e})".format(k, cvxsol[i][k], cvxsol[i-1][k], r))
#                 assert np.all(cvxsol[i][k] == cvxsol[i - 1][k])


def test2():
    prob, x_vars = example2()
    for i in range(100):
        compare_proxes(prob, x_vars, i)


def compare_proxes(prob, x_vars, i):
    """ Form the prox problem, put in some random data,
    and test that the output from solving with CVXPY/SCS
    is the same as solving with my custom proxer that skips CVXPY.

    Should have identical input data, so output should be *exactly* identical.
    """
    problem_data, indmap, solmap = stuffed_prox(prob, x_vars)
    pxprob, x0_vars = form_prox(prob, x_vars)

    # need to set x0 and rho
    rho = 1.0
    # rand_param_vals(x0_vars)
    x0_vars['x'].value = np.array([2, 3])
    x0_vars['y'].value = np.array([4, 5])
#    x0_vars['x'].value = np.array([-0.38079885, 0.91308715, -1.3308185])
#    x0_vars['y'].value = np.array([0.54995098, 1.62418463])
    x0_vars['__tau'].value = rho / 2

    pxprob.solve(solver=cp.SCS, verbose=True, warm_start=False)
    cvxsol = {}
    for k in x_vars:
        x = np.atleast_1d(np.squeeze(np.array(x_vars[k].value)))
        cvxsol[k] = x

    x0_vals = {}
    for k in x_vars:
        x0_vals[k] = np.atleast_1d(np.squeeze(np.array(x0_vars[k].value)))

    mysol = do_prox(problem_data, indmap, solmap, x0_vals, rho)

    print("\ni = {}".format(i))
    for k in mysol:
        r = cp.norm(mysol[k] - cvxsol[k]).value
        print("{}:\t{}\n\t{}  (r = {:.3e})".format(k, mysol[k], cvxsol[k], r))
        assert np.all(mysol[k] == cvxsol[k])

