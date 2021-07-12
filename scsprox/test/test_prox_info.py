from scsprox import Prox
from scsprox.examples import example


def test1():
    prob, xvars = example()
    prox = Prox(prob, xvars)

    x = prox()
    # Assert that info has correct keys.
    for key in 'status', 'iter', 'time':
        assert key in prox.info

    assert 'x' in x

    # Test that SCS settings get passed down.
    prox.update_settings(eps=4.5, max_iters=123)
    assert prox.settings['eps'] == 4.5
    assert prox.settings['max_iters'] == 123

    # Note: settings don't actually get passed to CySCS until we call the CySCS solve internally,
    # i.e., compute the prox.
    assert prox._work.settings['eps'] != 4.5
    assert prox._work.settings['max_iters'] != 123
    prox()
    assert prox._work.settings['eps'] == 4.5
    assert prox._work.settings['max_iters'] == 123
