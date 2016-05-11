from scs_prox.scs_mapping import get_solmap, extract_sol, dummy_scs_output, form_prox, rand_param_vals
from scs_prox.examples import example, example2, example3
import numpy as np

import scs

def compare_sols(prob, x_vars):
    # helper function could take in prob, x_vars
    # should work the same for proxed and unproxed function
    # oooh, what about tau?
    solmap = get_solmap(prob, x_vars)

    # get dummy SCS output
    data = prob.get_problem_data('SCS')
    out = dummy_scs_output(data)

    # make sure SCS.unpack results and our thing get the same answer
    mysol = extract_sol(out['x'], solmap)

    prob.unpack_results('SCS', out)
    cvxsol = {}
    for k in x_vars:
        x = np.atleast_1d(np.squeeze(np.array(x_vars[k].value)))
        cvxsol[k] = x

    for k in mysol:
        assert np.all(mysol[k] == cvxsol[k])

def test1():
    # test all three of the simple example problems
    for ex in example, example2, example3:
        prob, x_vars = ex()
        compare_sols(prob, x_vars)

def test2():
    for ex in example, example2, example3:
        prob, x_vars = ex()

        # create prox problem
        pxprob, x0_vars = form_prox(prob, x_vars)
        # need to initialize the x0 parameters so that CVXPY can parse
        rand_param_vals(x0_vars)

        compare_sols(pxprob, x_vars)





