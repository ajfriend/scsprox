from scsprox import CVXPYProx
from scsprox.examples import example

def test1():
    prob, xvars = example()
    prox = CVXPYProx(prob, xvars)
    prox()