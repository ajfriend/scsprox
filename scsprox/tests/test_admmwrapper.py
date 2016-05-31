from scsprox import Prox, admmwrapper
from scsprox.examples import example

def test1():
    prob, xvars = example()
    proxobj = Prox(prob, xvars)
    proxfunc = admmwrapper(proxobj)

    x, info = proxfunc()
    # assert that info has correct keys
    for key in 'status', 'iter', 'time':
        assert key in info

    assert 'x' in x

    # test that SCS settings get passed down
    proxfunc(eps=4.5, max_iters=123)

    assert proxobj._work.settings['eps'] == 4.5
    assert proxobj._work.settings['max_iters'] == 123