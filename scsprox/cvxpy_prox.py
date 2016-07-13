

class CVXPYProx(object):
    def __init__(self, prob, x_vars, **settings):
        """ Forms the proximal problem, stuffs the appropriate SCS matrices,
        and stores the array/matrix data.
        After initialization, doesn't depend on CVXPY in any way.

        Parameters
        ----------
        prob: CVXPY problem
        x_vars: dict
            Dict of the CVXPY Variables we want to prox on. Keys give the names
            of the variables as they'll be referred to in the input to the prox

        """

    def __call__(self, x0=None, rho=1.0, **settings):
        #return self._do(x0, rho, **settings)
        pass

    @staticmethod
    def form_prob(self, prob, xvars, rho):
        x0_vars = {}
        obj = 0
        for k in xvars:
            x = xvars[k]
            x0 = cvx.Parameter(*x.size)
            x0_vars[k] = x0
            
            obj = obj + rho/2.0*cvx.sum_squares(x - x0)
        
        pxprob = cvx.Problem(prob.objective + cvx.Minimize(obj), prob.constraints)
        
        return pxprob, x0_vars