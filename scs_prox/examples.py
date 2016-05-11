import cvxpy as cvx


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