import numpy as np

def test_1():
    import scs_prox
    prob, x_vars = scs_prox.examples.example()

    pxprob, x0_vars = scs_prox.scs_mapping.form_prox(prob, x_vars)

    x = 3
    rho = 2 # x should decrease by 1/rho

    x0_vars['x'].value = x    
    x0_vars['__tau'].value = rho/2.0

    pxprob.solve()

    assert np.allclose(x_vars['x'].value, x-1.0/rho)

