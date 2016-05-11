from scs_prox.prox_obj import Prox
from scs_prox.examples import example, example2, example3


def test():
    for ex in example, example2, example3:
        prob, x_vars = ex()
        prox = Prox(prob, x_vars)

        x_vals = prox.prox()
        print(x_vals)