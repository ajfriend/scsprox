from scsprox import Prox
from scsprox.examples import example

def test1():
    prob, xvars = example()
    prox = Prox(prob, xvars)

    x = prox()
    # assert that info has correct keys
    for key in 'status', 'iter', 'time':
        assert key in prox.info

    assert 'x' in x

    # test that SCS settings get passed down
    prox(eps=4.5, max_iters=123)

    assert prox._work.settings['eps'] == 4.5
    assert prox._work.settings['max_iters'] == 123