
import numpy as np
import cyscs

from .scsprox import stuffed_prox, do_prox_work
from .timer import DictTimer

class Prox:
    """ Class which forms the prox problem for a given CVXPY problem and variables.

    SCS input data is stuffed only once at initialization to save time.
    Class takes care of re-stuffing data for new x0 prox input,
    and uses CySCS to cache the SCS matrix factorization for speed.

    The class also automatically warm-starts the SCS solver based on
    the previous solution.

    SCS options can be passed in as key-word arguments to the init
    or the .do() function. This might set the max iters or the solver tolerance.

    Solver info can be seen from the prox.info attribute.

    """
    def __init__(self, prob, x_vars, verbose=False, **kwargs):
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
        kwargs['verbose'] = verbose

        data, self._indmap, self._solmap = stuffed_prox(prob, x_vars)

        self._work = cyscs.Workspace(data, data['dims'], **kwargs)

        self._bc = dict(b=data['b'],c=data['c'])

        self._warm_start = None

    @property
    def info(self):
        info = {}
        # convert to seconds
        info['setup_time'] = self._work.info['setupTime']*1e-3
        info['solve_time'] = self._work.info['solveTime']*1e-3
        info['iter'] = self._work.info['iter']
        info['status'] = self._work.info['status']

        return info

    @property
    def zero_elem(self):
        """ Using the names and sizes of the input variables,
        construct and return the zero element for this proxer.
        """
        x0 = {}
        for k in self._solmap:
            s = self._solmap[k] # a slice object
            length = s.stop - s.start
            if length == 1:
                x0[k] = 0.0
            elif length > 1:
                x0[k] = np.zeros(length)
            else:
                raise ValueError('Solmap must contain nonzero slices.')

        return x0

    def reset_warm_start(self):
        self._warm_start = None


    def do(self, x0=None, rho=1.0, **kwargs):
        """ Do the prox computation based on values in `x0`.
        `x0` can be None or an empty dict, in which case, it will prox
        on the 0 element of the appropriate size.
        """

        if not x0:
            x0 = self.zero_elem

        x, scs_sol = do_prox_work(self._work, self._bc, self._indmap,
                                  self._solmap, x0, rho,
                                  warm_start=self._warm_start, **kwargs)

        self._warm_start = dict(x=scs_sol['x'], y=scs_sol['y'], s=scs_sol['s'])

        if 'Solved' not in self.info['status']:
            msg = 'Unexpected solver status: {}'.format(self.info['status'])
            raise RuntimeError(msg)

        return x

def admmwrapper(prox):
    """ Form a prox function that returns an `info` dict along with prox vars.
    This is used for ADMM algorithms which require pure prox functions
    (not objects), and expect state information to be returned along with
    prox variables.
    """
    def foo(x0=None, rho=1.0, **kwargs):
        """ ADMM-style pure prox function.

        Parameters
        ----------
        x0: dict
            Prox input variables
        rho: float
            prox rho value
        kwargs: dict
            Optional SCS solver settings

        Returns
        -------
        x: dict
            Result of the prox computation
        info: dict
            Solver status information, with keys: 
            'time', 'iter', 'status'
        """
        x = prox.do(x0, rho, **kwargs)
        
        info = prox.info
        info = dict(time=info['solve_time'],
                    iter=info['iter'],
                    status=info['status'])

        return x, info
    
    return foo
