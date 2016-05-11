import scs
import numpy as np

from scs_prox.scs_mapping import get_solmap, extract_sol, form_prox, rand_param_vals, param_map, restuff
from scs_prox.examples import example, example2, example3


"""
Note: only works with x_vars pointing to scalars or vectors,
but not matrices

"""

def stuffed_prox(prob, x_vars):
    pxprob, x0_vars = form_prox(prob, x_vars)

    data, indmap = param_map(pxprob, x0_vars)
    solmap = get_solmap(pxprob, x_vars, data=data)
    
    return data, indmap, solmap

def do_prox(data, indmap, solmap, x0_vals, rho):    
    # don't modify original dict
    x0_vals = dict(x0_vals)
    # set tau in x0_vals
    x0_vals['__tau'] = rho/2.0
    
    restuff(data, indmap, x0_vals)
    
    out = scs.solve(data, data['dims'], verbose=False)
    scs_x = out['x']
    
    x_vals = extract_sol(scs_x, solmap)
    
    return x_vals