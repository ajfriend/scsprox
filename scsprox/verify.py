import numpy as np

# check that everything else stayed the same...
def check(data, x0_vars, indmap):
    assert np.all(x0_vars['__tau'].value == data['c'][indmap['__tau']])
    
    for k in x0_vars:
        if k != '__tau':
            x = x0_vars[k]
            x = np.array(-2*x.value).flatten()
            assert np.all(x == data['b'][indmap[k]])

def check_indmap(pxprob, x0_vars, indmap):
    # i think we can greatly simplify this testing when we can stuff
    rand_param_vals(x0_vars)
    pxprob = copy_prob(pxprob)
    data0 = pxprob.get_problem_data('SCS')
    
    check(data0, x0_vars, indmap)
    
    # change the x0 variables
    rand_param_vals(x0_vars)
    
    x0_vals = {k:np.array(v.value).flatten() for k,v in x0_vars.items()}
    
    pxprob = copy_prob(pxprob)
    restuff(data0, indmap, x0_vals)
    
    data1 = pxprob.get_problem_data('SCS')    
    check(data1, x0_vars, indmap)
    
    
    # check that the two data dicts turn out to be the same.
    # confirms our custom stuffing matches CVXPY stuffing
    assert (data0['A'] != data1['A']).nnz == 0
    assert data0['dims'] == data1['dims']
    
    for k in 'c', 'b':
        assert np.all(data0[k] == data1[k])
        # make sure not actually the same memory location
        assert data0[k] is not data1[k]
    
    
    return data0, data1

