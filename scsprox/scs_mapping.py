import cvxpy as cp
import numpy as np

from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict


def copy_prob(prob):
    return cp.Problem(prob.objective, prob.constraints)


def get_scs_data(prob):
    problem_data = ProblemData(*prob.get_problem_data(cp.SCS))

    # convert sparse indices and pointers to int64 so that CySCS doesn't
    # need to convert them itself (which raises a warning).
    A = problem_data.data['A']
    A.indptr = A.indptr.astype(np.int64)
    A.indices = A.indices.astype(np.int64)

    return problem_data


def form_prox(prob, x_vars):
    """ Given a CVXPY problem, form its prox.
    
    Given problem: min f(x),
    
    form the prox problem:

    min f(x) + tau * ||x - x0||^2
    
    Parameters
    ----------
    prob: CVXPY problem
    x_vars: dict
        Dict of k:v pairs, where k is a string, and v is a CVXPY Variable that appears in `prob`
        
    Returns
    -------
    pxprob: CVXPY prox problem
    x0_vars: dict
        Dict of k:v pairs, where v is CVXPY Parameter object, corresponding to prox input x0
        x0_vars also contains special key '__tau', which corresponds to the regularization parameter
    """

    tau = cp.Parameter(nonneg=True)
    x0_vars = {'__tau': tau}

    obj = 0
    for k, x in x_vars.items():
        x0 = cp.Parameter(shape=x.shape)
        x0_vars[k] = x0
        
        obj = obj + tau * cp.sum_squares(x - x0)
    
    pxprob = cp.Problem(prob.objective + cp.Minimize(obj), prob.constraints)
    
    return pxprob, x0_vars


def rand_param_vals(x0_vars):
    """ Set the CVXPY Parameter values for the x0 prox input parameters to random numpy
    arrays of the appropriate size.
    
    We use this to detect where the parameters get mapped to in the SCS stuffing.
    
    Modifies the x0_vars in place!
    """
    for k, x in x0_vars.items():
        if k == '__tau':
            x.value = max(np.random.standard_normal() + 1, 0)
        else:
            if x.ndim == 0:
                # I think it's better to force it to be a scalar, to be consistent
                # with the setting of tau.
                x.value = np.random.standard_normal()
            else:
                x.value = np.random.standard_normal(size=x.shape)
            
            
def param_map(pxprob, x0_vars):
    """
    get the location for taus and x0 parameter input
    in terms of the stuffed SCS problem
    
    Parameters
    ----------
    prox_prob: CVXPY prox problem
    x0_vars: dict
        dictionary of x0 keys and __tau that point to cvxpy Parameter objects
        
    Returns
    -------
    dict
        mapping of variable names (including '__tau') to indices of the b, c vectors in
        stuffed SCS problem
    
    Notes
    -----
    xs get multiplied by -2.
    expect '__tau' to be an array of random locations to c
    expect the other elements to be slice objects of continuous chunks
    """
    # set the x0 parameters to random values, and then find those random values in the vectors
    rand_param_vals(x0_vars)
    
    problem_data = get_scs_data(pxprob)
    
    # tau may be mapped to multiple locations in the c vector
    taus, = np.where(problem_data.data['c'] == x0_vars['__tau'].value)
    
    indmap = {'__tau': taus}
    
    b = problem_data.data['b']
    for k in x0_vars:
        if k != '__tau':
            # because x.value sometimes returns a float, and sometimes returns a 2d np.Matrix...
            # this cleans it all up to be a 1d array in either case
            x = np.atleast_1d(np.squeeze(np.array(x0_vars[k].value)))
            ind, = np.where(b == -2 * x[0])
            ind = ind[0]
            
            indmap[k] = slice(ind, ind + len(x))
    
    return problem_data, indmap


def restuff(data, indmap, x0_vals):
    """ Modify the b,c data in `data` to reflect the x0 prox values
    in x0_vals (which should be appropriately sized numpy arrays or scalars).

    data:
    indmap: is the mapping from variable names to b,c indices
    x0_vals:
    
    Notes
    -----
    tau maps to tau in c, but x maps to -2*x in b.
    """
    
    c = data['c']
    c[indmap['__tau']] = x0_vals['__tau']
    
    b = data['b']
    for k in x0_vals:
        if k != '__tau':
            b[indmap[k]] = -2*x0_vals[k]


def dummy_scs_output(data):
    """ `data` is a dict of SCS input data
    """
    x = np.random.randn(*data['c'].shape)
    y = np.random.randn(*data['b'].shape)

    out = {'info': {'dobj': 0,
                    'iter': 0,
                    'pobj': 0,
                    'relGap': 0,
                    'resDual': 0,
                    'resInfeas': 0,
                    'resPri': 0,
                    'resUnbdd': 0,
                    'setupTime': 0,
                    'solveTime': 0,
                    'status': 'Solved',
                    'statusVal': 1},
           'x': x,
           'y': y,
           's': y}

    return out


def get_solmap(prob, x_vars, problem_data=None):
    """
    Parameters
    ----------
    prob:
    x_vars: dict
        k:v pairs, where v is a CVXPY Variable
    problem_data:

    Returns
    -------
    dict
        elements are Python `slice` objects.
    """
    if problem_data is None:
        problem_data = get_scs_data(prob)
    
    out = dummy_scs_output(problem_data.data)
    prob.unpack_results(out, problem_data.solving_chain, problem_data.inverse_data)
    
    solmap = {}
    for k in x_vars:
        x = np.atleast_1d(np.squeeze(np.array(x_vars[k].value)))
        ind, = np.where(x[0] == out['x'])
        ind = ind[0]
        solmap[k] = slice(ind, ind+len(x))
        
    return solmap


def extract_sol(scs_x, solmap):
    """ Extract a solution from the SCS output variable `x`.
    solmap is a dict mapping variable names to indices (slices) of x.
    """
    x_vals = {}
    for k in solmap:
        x_vals[k] = scs_x[solmap[k]]
        if len(x_vals[k]) == 1:
            x_vals[k] = x_vals[k][0]
        
    return x_vals


class ProblemData:
    def __init__(self, data, solving_chain, inverse_data):
        self.data = data
        self.solving_chain = solving_chain
        self.inverse_data = inverse_data

    @property
    def data(self):
        return self._data

    @property
    def solving_chain(self):
        return self._solving_chain

    @property
    def inverse_data(self):
        return self._inverse_data

    @data.setter
    def data(self, value):
        self._data = value

    @solving_chain.setter
    def solving_chain(self, value):
        self._solving_chain = value

    @inverse_data.setter
    def inverse_data(self, value):
        self._inverse_data = value

    @property
    def cone_dims_for_scs(self):
        return dims_to_solver_dict(self._data['dims'])
