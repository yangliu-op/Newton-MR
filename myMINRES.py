# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:48:45 2020

MINRES with NPC detection

@author: Yang Liu
"""

import torch

def solvetm(A, b, xl, rtl, Delta, xnorm, phil, m_r, iters):    
    xTr = torch.dot(xl, rtl)
    diff = Delta**2 - xnorm**2
    omega = (-xTr + torch.sqrt(xTr**2 + phil**2*diff))/phil**2
    x = xl + omega*rtl   
    m_x = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
    iters = iters + 1
    if m_x <= m_r:
        return x, m_x, iters
    else:
        return Delta*rtl/phil, m_r, iters


def lanczos(A, v, vp, beta, shift=0):
    Av = Ax(A, v)
    
    if shift == 0:
        pass
    else:
        Av = Av + shift*v
    
    alfa = torch.dot(Av, v)
    vn = Av - v*alfa - vp*beta   
    
    betan = torch.norm(vn)
    vn = vn/betan

    return vn, v, alfa, betan

def qrdecomp(alfa, betan, delta1, sn, cs):
    # delta1n = delta1
    delta2 = cs*delta1 + sn*alfa
    gamma1 = sn*delta1 - cs*alfa
    epsilonn = sn*betan
    delta1n = -cs*betan
    return delta2, epsilonn, delta1n, gamma1

def updates(gamma2, cs, sn, phi, vn, v, wl, wl2, epsilon, delta2, xl, rtl):
    tau = cs*phi
    phi = sn*phi

    w = (v - epsilon*wl2 - delta2*wl)/gamma2
    x = xl + tau*w
    rt = sn**2 * rtl - phi * cs * vn
    return x, w, wl, rt, tau, phi


def myMINRES(A, b, rtol, maxit, shift=0,
             reOrth=True, isZero=1E-15):
    """    
    minres with indefiniteness detector of solving min||b - Ax||
    
    Inputs,
    rtol: inexactness control
    maxit: maximum iteration control.
    shift: perturbations s.t., solving min||b - (A + shift)x||
    detector: bool input, control whether non-positive curvature retur.n or not.
    reOrth: bool input, control whether to apply reorthogonalisation or not.
    isZero: decide a number should be regard as 0 if smaller than isZero.
    
    Output flag explaination,
    Flag = 1, inexactness solution.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    Flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    """        
    
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    delta1n = 0
    epsilonn = 0
    gamma2 = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rt = b 
    vn = r3/betan
    dType = 'Sol'
    # print((A @ b).norm())
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    v = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 9        
        
    if reOrth:
        V = vn.reshape(-1, 1)
        
#    normHy2 = tau**2
    while flag == flag0:
        #lanczos
        vn, v, alfa, betan = lanczos(A, vn, v, betan, shift)
        iters += 1
        if reOrth:
            if iters == 1:
                vn = vn - (v @ vn)*v
            else:
                vn = vn - torch.mv(V, torch.mv(V.T, vn))
        
        if reOrth:
            V = torch.cat((V, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        delta1 = delta1n
        epsilon = epsilonn
        delta2, epsilonn, delta1n, gamma1 = qrdecomp(alfa, betan, delta1, sn, cs)
        
        csl = cs
        phil = phi
        rtl = rt
        xl = x  
        nc = csl * gamma1         
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gamma2 = symGivens(gamma1, betan) 
        
        if phil*torch.sqrt(gamma2**2 + delta1n**2) < rtol*torch.sqrt(beta1**2-phil**2):
            flag = 1  ## trustful least square solution
            return xl, relres, iters, rtl, dType
        
        if nc >= 0: # NPC detection
            flag = 3
            dType = 'NC'
            return xl, relres, iters, rtl, dType        
        
        if gamma2 > isZero:
            x, w, wl, rt, tau, phi = updates(gamma2, cs, sn, phi, vn, v, w, 
                                             wl, epsilon, delta2, xl, rtl)
        else:
            ## if gamma1 = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gamma2 = 0  
            phi = phil
            rt = rtl
            x = xl
            flag = 2
            print('flag = 2, b is not in the range(A)!')
            maxit += 1
            return x, relres, iters, rt, dType
            
        rnorm = phi
        relres = rnorm / beta1
        if iters >= maxit:
            flag = 4  ## exit before maxit
            dType = "MAX"
            # print('Maximun iteration reached', flag, iters)
            return x, relres, iters, rt, dType
    return x, relres, iters, rt, dType


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = torch.mv(A, x)
    return Ax

def precond(M, r):
    if callable(M):
        h = M(r)
    else:
        h = torch.mv(torch.pinverse(M), r)
    return h

def symGivens(a, b, device="cpu"):
    """This is used by the function minresQLP.
    
    Arguments:
        a (float): A number.
        b (float): A number.
        device (torch.device): The device tensors will be allocated to.
    """
    if not torch.is_tensor(a):
        a = torch.tensor(float(a), device=device)
    if not torch.is_tensor(b):
        b = torch.tensor(float(b), device=device)
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = torch.sign(a)
        s = 0
        r = torch.abs(a)
    elif a == 0:
        c = 0
        s = torch.sign(b)
        r = torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
        t = a / b
        s = torch.sign(b) / torch.sqrt(1 + t ** 2)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = torch.sign(a) / torch.sqrt(1 + t ** 2)
        s = c * t
        r = a / c
    return c, s, r

    
    
    
