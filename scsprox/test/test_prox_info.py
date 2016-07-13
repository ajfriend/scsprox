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
    prox.update_settings(eps=4.5, max_iters=123)
    assert prox.settings['eps'] == 4.5
    assert prox.settings['max_iters'] == 123

    # note: settings don't actually get passed to CySCS until we call the CySCS solve internally, i.e., compute the prox
    assert prox._work.settings['eps'] != 4.5
    assert prox._work.settings['max_iters'] != 123
    prox()
    assert prox._work.settings['eps'] == 4.5
    assert prox._work.settings['max_iters'] == 123