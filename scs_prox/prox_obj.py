
import numpy as np
import scs

from scs_prox.scs_prox import stuffed_prox, do_prox

# todo: warm-starting
# evaluate on a real problem to check that its doing what we want
# have it return info, or, at least, make info accessible.
# timing info?

class Prox:
    def __init__(self, prob, x_vars):
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
        self.data, self.indmap, self.solmap = stuffed_prox(prob, x_vars)

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


    def prox(self, x0=None, rho=1.0):
        """ Do the prox computation based on values in `x0`.
        `x0` can be None or an empty dict, in which case, it will prox
        on the 0 element of the appropriate size.
        """

        if not x0:
            x0 = self.zero_elem()

        # need some warm starting shit
        # make it so we can clear/set warm-start params

        # maybe go lower level
        x = do_prox(self.data, self.indmap, self.solmap, x0, rho)

        return x
