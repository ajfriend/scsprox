
import numpy as np
import cyscs

from scs_prox.scs_prox import stuffed_prox, do_prox_work
from .timer import DictTimer

# todo: warm-starting
# evaluate on a real problem to check that its doing what we want
# have it return info, or, at least, make info accessible.
# timing info?
# initial factorization timing, and last solve time

class Prox:
    def __init__(self, prob, x_vars, **kwargs):
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
        self.info = {}

        with DictTimer('stuffing_time', self.info):
            data, self.indmap, self.solmap = stuffed_prox(prob, x_vars)

        with DictTimer('init_time_outer', self.info):
            self.work = cyscs.Workspace(data, data['dims'], **kwargs)

        self.bc = dict(b=data['b'],c=data['c'])

        self._warm_start = None

        self.info['init_time_inner'] = self.work.info['setupTime']*1e-3 # convert to seconds

        for key in 'prox_time_outer', 'prox_time_inner', 'iter', 'status':
            self.info[key] = None

    def zero_elem(self):
        """ Using the names and sizes of the input variables,
        construct and return the zero element for this proxer.
        """
        x0 = {}
        for k in self.solmap:
            s = self.solmap[k] # a slice object
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


    def prox(self, x0=None, rho=1.0, **kwargs):
        """ Do the prox computation based on values in `x0`.
        `x0` can be None or an empty dict, in which case, it will prox
        on the 0 element of the appropriate size.
        """

        if not x0:
            x0 = self.zero_elem()

        with DictTimer('prox_time_outer', self.info):
            x, scs_sol = do_prox_work(self.work, self.bc, self.indmap,
                         self.solmap, x0, rho, warm_start=self._warm_start, **kwargs)

        self.info['prox_time_inner'] = self.work.info['solveTime']*1e-3
        self.info['iter'] = self.work.info['iter']
        self.info['status'] = self.work.info['status']

        self._warm_start = dict(x=scs_sol['x'], y=scs_sol['y'], s=scs_sol['s'])

        if 'Solved' not in self.info['status']:
            raise RuntimeError('Unexpected solver status: {}'.format(self.info['status']))


        return x
