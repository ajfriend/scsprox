import scs

from .scs_mapping import get_solmap, extract_sol, form_prox, rand_param_vals, param_map, restuff


"""
Note: only works with x_vars pointing to scalars or vectors,
but not matrices

"""


def stuffed_prox(prob, x_vars):
    pxprob, x0_vars = form_prox(prob, x_vars)

    problem_data, indmap = param_map(pxprob, x0_vars)
    solmap = get_solmap(pxprob, x_vars, problem_data=problem_data)
    
    return problem_data, indmap, solmap


def do_prox(problem_data, indmap, solmap, x0_vals, rho):
    # don't modify original dict
    x0_vals = dict(x0_vals)
    # set tau in x0_vals
    x0_vals['__tau'] = rho / 2.0

    data = problem_data.data
    restuff(data, indmap, x0_vals)

    # Solve via SCS directly.
    out = scs.solve(data, problem_data.cone_dims_for_scs, verbose=True)
    scs_x = out['x']
    
    x_vals = extract_sol(scs_x, solmap)
    
    return x_vals, out['info']


def do_prox_work(work, bc, indmap, solmap, x0_vals, rho, warm_start=None, **settings):
    # don't modify original dict
    x0_vals = dict(x0_vals)
    # set tau in x0_vals
    x0_vals['__tau'] = rho/2.0
    
    # modifies bc
    restuff(bc, indmap, x0_vals)
    
    scs_sol = work.solve(new_bc=bc, warm_start=warm_start, **settings)
    scs_x = scs_sol['x']
    
    x_vals = extract_sol(scs_x, solmap)
    
    return x_vals, scs_sol