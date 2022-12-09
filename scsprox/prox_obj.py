import numpy as np
import cyscs

from .scsprox import stuffed_prox, do_prox_work
from .timer import DictTimer

_cvxpytime = 'cvxpy_time'
_outer_setup_time = 'outer_scs_setup_time'
_scs_setup_time = 'scs_setup_time'


class Prox(object):
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

    def __init__(self, prob, x_vars, **settings):
        """ Forms the proximal problem, stuffs the appropriate SCS matrices,
        and stores the array/matrix data.
        After initialization, doesn't depend on CVXPY in any way.

        Parameters
        ----------
        prob: CVXPY problem
        x_vars: dict
            Dict of the CVXPY Variables we want to prox on. Keys give the names
            of the variables as they'll be referred to in the input to the prox0
        """
        self.settings = self.default_settings()
        self.update_settings(**settings)

        self._info = {}
        with DictTimer(_cvxpytime, self._info):
            problem_data, self._indmap, self._solmap = stuffed_prox(prob, x_vars)

        data = problem_data.data
        with DictTimer(_outer_setup_time, self._info):
            cone = problem_data.cone_dims_for_scs
            self._work = cyscs.Workspace(data, cone, **self.settings)

        self._bc = dict(b=data['b'], c=data['c'])
        self._warm_start = None

    def __call__(self, x0=None, rho=1.0, **settings):
        return self._do(x0, rho, **settings)

    @property
    def info(self):
        # convert to seconds
        info = {_scs_setup_time: self._work.info['setupTime'] * 1e-3,
                'time': self._work.info['solveTime'] * 1e-3,
                'iter': self._work.info['iter'],
                'status': self._work.info['status']}

        for k in _cvxpytime, _outer_setup_time:
            info[k] = self._info[k]

        return info

    @property
    def zero_elem(self):
        """ Using the names and sizes of the input variables,
        construct and return the zero element for this proxer.
        """
        x0 = {}
        for k in self._solmap:
            s = self._solmap[k]  # a slice object
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

    @staticmethod
    def default_settings():
        return dict(eps=1e-3, max_iters=100, verbose=False)

    def check_settings(self):
        default = self.default_settings()
        if not set(default.keys()) == set(self.settings.keys()):
            raise ValueError('Invalid settings dict. Compare with default_settings().')
            
    def update_settings(self, **settings):
        for key in settings:
            if key in self.settings:
                self.settings[key] = settings[key]
            else:
                raise ValueError('Invalid settings key: {}'.format(key))
                
        self.check_settings()

    def _do(self, x0=None, rho=1.0, **settings):
        """ Do the prox computation based on values in `x0`.
        `x0` can be None or an empty dict, in which case, it will prox
        on the 0 element of the appropriate size.
        """
        self.update_settings(**settings)

        if not x0:
            x0 = self.zero_elem

        x, scs_sol = do_prox_work(self._work, self._bc, self._indmap,
                                  self._solmap, x0, rho,
                                  warm_start=self._warm_start, **self.settings)

        self._warm_start = dict(x=scs_sol['x'], y=scs_sol['y'], s=scs_sol['s'])

        if 'Solved' not in self.info['status']:
            msg = 'Unexpected solver status: {}'.format(self.info['status'])
            raise RuntimeError(msg)

        return x
