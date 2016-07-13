import numpy as np
import cvxpy as cvx

def print_file():
    import os
    print('\nRunning package from:', os.path.abspath(__file__))


def example():
    x = cvx.Variable()

    obj = cvx.Minimize(x)

    cons = [x >= 0]

    prob = cvx.Problem(obj, cons)
    
    x_vars = dict(x=x)
    
    return prob, x_vars

def example2():
    x = cvx.Variable(3)
    y = cvx.Variable(2)

    prob = cvx.Problem(cvx.Minimize(cvx.norm(x) + cvx.norm(2*y)))
    x_vars = dict(x=x,y=y)
    
    return prob, x_vars
    
def example3():
    x = cvx.Variable(3)
    y = cvx.Variable()

    prob = cvx.Problem(cvx.Minimize(cvx.norm(x) + cvx.norm(2*y)))
    x_vars = dict(x=x,y=y)
    
    return prob, x_vars

def example_rand(m=10,n=5,seed=0):
    assert m > n
    np.random.seed(0)
    A = np.random.randn(m,n)
    b = np.random.randn(m)

    x = cvx.Variable(n)
    y = cvx.Variable(m)
    z = cvx.Variable()

    obj = cvx.sum_squares(A*x-b) + cvx.norm(A.T*y - x) + 0.1*cvx.norm(y) + cvx.norm(z-y)

    prob = cvx.Problem(cvx.Minimize(obj))
    x_vars = dict(x=x,y=y,z=z)

    prob.solve(solver='ECOS')
    true_sol = dict(x=x.value, y=y.value, z=z.value)
    for k in true_sol:
        true_sol[k] = np.array(true_sol[k]).flatten()
        if len(true_sol[k]) == 1:
            true_sol[k] = true_sol[k][0]

    return prob, x_vars, true_sol