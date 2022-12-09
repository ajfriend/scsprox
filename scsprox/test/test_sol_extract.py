import numpy as np
import cvxpy as cp

from scsprox.examples import example, example2, example3
from scsprox.scs_mapping import get_solmap, extract_sol, dummy_scs_output, form_prox, rand_param_vals, ProblemData


def compare_sols(prob, x_vars):
    solmap = get_solmap(prob, x_vars)

    # get dummy SCS output
    problem_data = ProblemData(*prob.get_problem_data(cp.SCS))
    out = dummy_scs_output(problem_data.data)

    # make sure SCS.unpack results and our thing get the same answer
    mysol = extract_sol(out['x'], solmap)

    prob.unpack_results(out, problem_data.solving_chain, problem_data.inverse_data)
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





