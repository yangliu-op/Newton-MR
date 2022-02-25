import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def linesearch(obj, fk, gTp, x, p, maxiter=200, alpha=1, c1=1e-4):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
    # fa = _fa(obj, x, alpha, p) 
    fa = obj(x + alpha * p, 'f')       
    # fk = obj(x, 'f')
    while fa > fk+alpha*c1*gTp and T < maxiter:
        alpha *= 0.5
        # fa = _fa(obj, x, alpha, p)  
        fa = obj(x + alpha * p, 'f')   
#        print(fa)
        T += 1    
        if alpha < 1E-18: 
            alpha = 0 # Kill small alpha
            break
    x = x + alpha*p    
    return x, alpha, T

def linesearchgrad(obj, gk2, Hg, x, p, maxiter=200, alpha=1, c1=1e-4):
    """    
    INPUTS:
        g: a function handle of gradient
        H: Hessian Hk vector product
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0    
    # ga2 = _ga2(obj, x, alpha, p)
    _, ga = obj(x + alpha * p, 'fg')   
    # gk = obj(x, 'g')
    while (ga.norm())**2 > gk2 + 2*alpha*c1*torch.dot(
            p, Hg) and T < maxiter:
        alpha *= 0.5
        _, ga = obj(x + alpha * p, 'fg')   
        # ga2 = _ga2(obj, x, alpha, p)
        T += 1       
        if alpha < 1E-18: 
            alpha = 0 # Kill small alpha
            break
    x = x + alpha*p    
    return x, alpha, T

# @profile
def linesearch_NewtonMR(obj, fk, gTp, x, p, maxiter=200, c1=1e-4):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
#    alpha = 2*(1 - 2*c1)
    alpha = 1
#    if alpha == 1:
    zeta = 2
#    else:
#        zeta = 2*(1 - 2*c1)
    # fa = _fa(obj, x, alpha, p) 
    # fa = obj(x + alpha * p, 'f')
    fa = obj(x + alpha * p, 'f')
    # fk = obj(x, 'f')
    # print(gTp, fa, fk)
    while fa > fk+alpha*c1*gTp and T < maxiter or torch.isnan(fa):
        alpha = alpha/zeta
        # fa = _fa(obj, x, alpha, p) 
        fa = obj(x + alpha * p, 'f')
        T = T + 1    
        if alpha < 1E-18: 
            alpha = 0 # Kill small alpha
            break
    x = x + alpha*p    
    return x, alpha, T

def linesearch_NewtonMR_NC(obj, fk, gTp, x, p, maxiter=200, c1=1e-4, alpha_max=1e8):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
#    alpha = 2*(1 - 2*c1)
    alpha = 1
    zeta = 2
    # fa = _fa(obj, x, alpha, p)
    fa = obj(x + alpha * p, 'f')   
    # fk = obj(x, 'f')
    # print(fa)
    if torch.isnan(fa):
        print('FisNan')
    if fa <= fk+alpha*c1*gTp:
#        print('NC Linesearch passes')
#        fal = fk
        # while fa <= fk+alpha*c1*gTp and T < maxiter and alpha < alpha_max:
        while fa <= fk+alpha*c1*gTp and T < maxiter and alpha <= alpha_max:
#            print('fa', fa, alpha)
            alpha = alpha*zeta
#            fal = fa
            # fa = _fa(obj, x, alpha, p) 
            fa = obj(x + alpha * p, 'f')   
            T = T + 1    
        x = x + alpha/zeta*p    
        return x, alpha, T
    else:
        while fa > fk+alpha*c1*gTp and T < maxiter or torch.isnan(fa):
#            print('fa', fa, alpha)
            alpha = alpha/zeta
            # print(alpha, zeta)
            # fa = _fa(obj, x, alpha, p)
            fa = obj(x + alpha * p, 'f')   
            T = T + 1    
            if alpha < 1E-18: 
                alpha = 0 # Kill small alpha
                break
        x = x + alpha*p    
        return x, alpha, T

def linesearchNC(obj, fk, x, p, maxiter=200, alpha=1, c1=1e-4):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """

    T = 0
    theta = 0.5
    # fa = _fa(obj, x, alpha, p)
    fa = obj(x + alpha * p, 'f')   
    # fk = obj(x,'f')
    # print(' p',p)
    while fa > fk - alpha**3*c1*p.norm()**3/6 and T < maxiter or torch.isnan(fa):
        alpha = alpha*theta
        # fa = _fa(obj, x, alpha, p)
        fa = obj(x + alpha * p, 'f')   
        T += 1    
        if alpha < 1E-18: 
            alpha = 0 # Kill small alpha
            break
    x = x + alpha*p
#    print(fa - fk)
    return x, alpha, T


# @profile
def linesearchzoom(obj, f0, g0p, x, p, maxiter=200, c1=1e-4, c2=0.4, amax=100, fe=1000):
    """    
    All vector are column vectors.
    INPUTS:
        fg: a function handle of both objective function and its gradient
        x: starting x
        p: direction p
        maxiter: maximum iteration of line search with strong Wolfe
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
        alpha0: initial step-size
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    #phi(x) = f(x), phi'(x) = g(x).T.dot(p)
    p = p.detach()
    itrs = 0
    itrfe = 0
    itrs2 = 0
    a1 = 0
    a2 = 1
    # f0, g0 = obj(x, 'fg')
    fb = f0 # f preview
    alpha = 0
    while itrs < maxiter:
        fa, ga = obj(x + a2*p, 'fg')
        # fa, ga = _fga(obj, x, a2, p)
        itrs += 1 ## call fa
#        print(f0, fa)
        if fa > f0 + a2*c1*g0p or (fa >= fb and itrs > 0):
            alpha, itrs2, itrfe = zoomf(a1, a2, obj, f0, fa, g0p, x, p, c1, c2, 
                                 itrs, itrfe, maxiter, fe)
            break
        itrfe += 2 ## call ga
        gap = torch.dot(ga, p)
        if abs(gap) <= -c2*g0p:
            alpha = a2
            itrs2 = 0
            break
        if gap >= 0:            
#            print('Zoom backward')
            alpha, itrs2, itrfe = zoomf(a2, a1, obj, f0, fa, g0p, x, p, c1, c2, 
                                 itrs, itrfe, maxiter, fe)
            # alpha, itrs2 = zoomf(a1, a2, obj, f0, fa, g0p, x, p, c1, c2, itrs, maxiter)
            break
        # if amax == 1:
        a2 = a2*2
        # else:
        #     a2 = min(a2*2, amax)
        fb = fa
        # print(a2)
        # itrs += 1
    itrs += itrs2
#     if itrs >= maxiter:        
#         alpha = 0
    x = x + alpha*p    
    if torch.is_tensor(alpha):
        alpha = alpha.item()  
    return x, alpha, itrs, itrfe

# @profile
def zoomf(a1, a2, obj, f0, fa, g0p, x, p, c1, c2, itrs, itrfe, maxiter, fe):
    itrs2 = 0
    aj = a1
    while (itrs2 + itrs) < maxiter: # use maxiter to evaluate Oracle calls
        #quadratic
        # lower bound
        fa1, ga1 = obj(x + a1*p, 'fg')
        # fa1, ga1 = _fga(obj, x, a1, p)
        itrs2 += 1 # call fa1, ga1
        itrfe += 2
        ga1p = torch.dot(ga1, p)
        # upper bound
        fa2, ga2 = obj(x + a2*p, 'fg')
        # fa2, ga2 = _fga(obj, x, a2, p)
        itrfe += 2 # call fa2, ga2
        while torch.isnan(fa2):
            # sometime even alpha_max=1 is too large
            a2 = a2/10
        ga2p = torch.dot(ga2, p)
        aj = cubicInterp(a1, a2, fa1, fa2, ga1p, ga2p)
        a_mid = (a1 + a2)/2
        if not inside(aj, a1, a_mid):
            aj = a_mid
        # faj, gaj = _fga(obj, x, aj, p)
        faj, gaj = obj(x + aj*p, 'fg')
        itrfe += 2 # call faj, gaj
        # print('f', faj, f0, a1, a2, aj)
        if faj > f0 + aj*c1*g0p or faj >= fa1:
            a2 = aj
        else:
            gajp = torch.dot(gaj, p)
#            print('g', gajp, -c2*g0p)
            if abs(gajp) <= -c2*g0p:
                break
            if gajp*(a2 - a1) >= 0:
                a2 = a1
            a1 = aj
        if abs(a1 - a2) < 1e-18 or aj < 1e-18:
            print("Warning: Strong Wolfe line-search fail to find a proper step-size.")
            print('a_low, a_high, a_i=', a1, a2, aj)
            aj = 0
            itrfe = fe
            break
    return aj, itrs2, itrfe

def cubicInterp(x1, x2, f1, f2, g1, g2):
    """
    find minimizer of the Hermite-cubic polynomial interpolating a
    function of one variable, at the two points x1 and x2, using the
    function (f1 and f2) and derivative (g1 and g2).
    """
#    print(x1, x2)
    # Nocedal and Wright Eqn (3.59)
    d1 = g1 + g2 - 3*(f1 - f2)/(x1 - x2)
    d2 = torch.sign(torch.tensor(float(x2 - x1)))*torch.sqrt(
        torch.tensor(float(d1**2 - g1*g2)))
    xmin = x2 - (x2 - x1)*(g2 + d2 - d1)/(g2 - g1 + 2*d2);
    return xmin
    
def inside(x, a, b):
    """
    test x \in (a, b) or not
    """
    l = 0
    # if not torch.isreal(x):
    #     return l
    if a <= b:
        if x >= a and x <= b:
            l = 1
    else:
        if x >= b and x <= a:
            l = 1
    return l

def Ax(A, v):
    if callable(A):
        Ax = A(v)
    else:
        Ax =torch.mv(A, v)
    return Ax

def tAx(A, v, epsilon):
    if callable(A):
        Ax = A(v) + 2*epsilon*v
    else:
        Ax = torch.mv(A + 2*epsilon*torch.eye(len(v)),v)
    return Ax