# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:48:45 2020

minres-type methods with indefiniteness detector of solving min||b - Ax||

Sometimes Lanczos contains large error and fail to return a beta_{r+1} close
to zero, then isZero could be set as 1E-7 or larger.

No preconditioner supported

@author: Yang Liu
"""

import torch
import numpy as np
from scipy.linalg import norm, inv, eig
from scipy.sparse.linalg import cg
        
#global mydtype
#mydtype = torch.float64

def solvetm(A, b, xl, rkl, Delta, xnorm, phil, m_r, iters):    
    xTr = torch.dot(xl, rkl)
    diff = Delta**2 - xnorm**2
    omega = (-xTr + torch.sqrt(xTr**2 + phil**2*diff))/phil**2
    x = xl + omega*rkl   
    m_x = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
    iters = iters + 1
    if m_x <= m_r:
        return x, m_x, iters
    else:
        return Delta*rkl/phil, m_r, iters

# def tm(rk, x=None):
#     if x is None:
#         tm = -
# @profile
def main(): 
    """Run an example of minresQLP."""
    from time import time
    torch.manual_seed(11167)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    n = 11
    A = torch.randn(n, n, device=device, dtype=torch.float64)
#    A = torch.eye(n, dtype=torch.float64)
#    A[-1, -1] = -1
#    A = torch.mm(A.T, A)
    b = torch.ones(n, device=device, dtype=torch.float64)
#    b[0] = b[0] + 1E-5
#    print(b)
#    x = torch.dot(b,b)
#    x = torch.mm(b.T, b)
#    C = A + A.T
    C = A + A.T
    A = torch.randn(n, n, dtype=torch.float64)
    B = A + A.T
#    A = torch.diag(torch.ones(n, dtype=torch.float64))
#    A[0,1] = 2
    B = A + A.T
    U, s, V = torch.svd(B)
#    ss = abs(0,3,n)
#    ss = torch.logspace(1,0,n, dtype=torch.float64)
    ss = torch.randn(n, dtype=torch.float64)
    ss = abs(ss)
#    ss[0] = 1E-2
#    ss[2] = 1E3
    d = 8
#    ss[d-1] = -1
    ss[d:] = 0
    print(ss)
    E = U @ torch.diag(ss) @ U.T
    xx = myMINRES_naive(E, b, 1E-8, d)
    print(' ')
#    Pb = U[:,:d] @ U[:,:d].T @ b
#    xx = myMINRES_naive(E, Pb, 1E-8, d)
#    print(' ')
    
#    xx = myMINRES_naive(E @ E, E @ b, 1E-8, d)
    # print(xx)
    t1 = time()
    
    
def myMINRES_naive(A, b, rtol, maxit, shift=0, reOrth=True, isZero=1E-15,L=1E2):
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
    Flag = -1, b and x are eigenvectors.
    Flag = 0, the exact solution is x = 0.
    Flag = 1, min-length solution for the LS problem, given rtol.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    
    If Flag = 0, then rank(A) = iters - 1.
    """        
    
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
    
    # rtol = np.sqrt(rtol)
    
#    if M is None:
#        noprecon = True
#        pass
#    else:
#        noprecon = False
#        r3 = preCond(M, r2)
#        beta1 = r3.T.dot(r2)
#        if beta1 < 0:
#            print('Error: "M" is indefinite!')
#        else:
#            beta1 = np.sqrt(beta1)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    dltan = 0
    eplnn = 0
    gama = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    LSType = 'one'
    dType = 'Sol'
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
#        Rn = rk.reshape(-1, 1)
    sumtau = 0
    
#    L = 1E2
#    d_gd = b
#    x_gd = d_gd /L 
#    r_gd = b - Ax(A, x_gd)
#    # if b.T.dot(Ax(A, b)) <= 0: # covered
#    #     flag = 3
#    #     #print('ncb', -nc)
#    #     return b, relres, iters
#        
##    normHy2 = tau**2
#    delta = beta1**2
#    p = b
#    r_cg = b
#    x_cg = torch.zeros(n,device=device, dtype=torch.float64)
    while flag == flag0:
#        Ap = Ax(A, p)
#        pAp = torch.dot(p, Ap)
#        alpha = delta/pAp
#        x_cg = x_cg + alpha*p
#        r_cg = r_cg - alpha*Ap
#        prev_delta = delta
#        delta = torch.dot(r_cg, r_cg)
#        p = r_cg + (delta/prev_delta)*p
##        m_cg = - x_cg @ ( r_cg + b) /2
#        m_cg = x_cg @ Ax(A, x_cg)/2 - b@x_cg
#        
#        m_gd = - x_gd @ ( r_gd + b) /2
#        d_gd = r_gd
#        x_gd = x_gd + d_gd/L
#        r_gd = b - Ax(A, x_gd)
        #lanczos
        betal = beta
        beta = betan
        v = vn
#        print(v)
        r3 = Ax(A, v)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
        
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
#                    print('inner_flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
#                    print('inner_flag = -1')
                    break
                
            
        vn = r3/betan
#        print(vn @ v)
        if reOrth:
            UU, ss, VV = torch.svd(A @ Vn)
            PP = UU[:,:iters] @ UU[:,:iters].T
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
        
        nc = csl * gbar 
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
#        gama = torch.sqrt(gbar**2 + betan**2)
#        cs = gbar/gama
#        sn = betan/gama
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
#            print(x.norm())
            x = x + tau*w
            sumtau += tau
#            print('tau_sum', sumtau)
#            normHy2 = tau**2 + normHy2
#            print((beta1**2 - phi**2)/x.norm()**2, end=' ')
#            px = x - (x @ rk)/phi**2 * rk
#            print('b',b, Vn @ (Vn.T @ b))
#            print('rk', rk, b -  Vn @ (Vn.T @ b))
            rk = sn**2 * rkl - phi * cs * vn
#            xh = x - x@rk/rk.norm()**2*(rk)
#            print(' ', (PP @ xh - xh).norm())
#            Rn = torch.cat((Rn, rk.reshape(-1, 1)), axis=1)
#            Hrg = torch.mv(A, rk) - b
#            print('hrg', b - Ax(A, x), rk)
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gama = 0  
            phi = phil
            rk = rkl
            x = xl
#            print(Vn[:,:-1].T @ x)
            flag = 2
#            print('inner_b is not in the range(A)!', flag, iters, relres)
            maxit += 1
#        print(' ', (rk @ x)/phi**2)
            
        xh = x-(rk @ x)/phi**2*rk
        phih = (b - Ax(A, xh)).norm()
#        print('xh',  ((xh @ (Ax(A, xh)))/2 - b@xh).item(), ((x @ (Ax(A, x)))/2 - b@x
#                      ).item(), phi.item(), phih.item())
#        m = - x @ ( rk + b) /2
#        print('MR', m, 'CG', m_cg, 'gd', m_gd)
#        print('diff', m_gd - m)
#        print('step_MR:', tau, 'step)
        ## stopping criteria
        xnorm = torch.norm(x)   
#        print(iters, gama, epln, dlta, xnorm)
#        if iters > 13:
#            print(' ')
#        print('angle', (x @ rk)**2/phi**2/xnorm**2 )
#        print(iters, xnorm**2/2/1E3)
        rnorm = phi
#        print(Rn.T @ (A @ Rn))
        relres = rnorm / beta1
#        print(iters, relres)
        # m0 = -x.T.dot(b) + x.T.dot(Ax(A,x))/2
        if relres <= rtol:
            return x, relres, iters, rk, phi
#        print(iters, maxit)
        if iters >= maxit:
            flag = 4  ## exit before maxit
#            print('inner_Maximun iteration reached', flag, iters, relres)
            return x, relres, iters, rk, phi
    return x, relres, iters, rk, phi
#    
def myMINRES(A, b, rtol, maxit, shift=0, psi=1E-12,
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
    Flag = -1, b and x are eigenvectors.
    Flag = 0, the exact solution is x = 0.
    Flag = 1, min-length solution for the LS problem, given rtol.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    
    If Flag = 0, then rank(A) = iters - 1.
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
    dltan = 0
    eplnn = 0
    gama = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    dType = 'Sol'
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 9        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
#    normHy2 = tau**2
    while flag == flag0:
        #lanczos
        betal = beta
        beta = betan
        v = vn
#        print(v)
        r3 = Ax(A, v)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
        
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
                    print('flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
                    print('flag = -1')
                    break
                
            
        vn = r3/betan
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
        
        nc = csl * gbar 
        
        if nc >= -isZero:
            flag = 3
            dType = 'NC'
            return rkl, relres, iters, rkl, dType
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            rk = sn**2 * rkl - phi * cs * vn
#            Hrg = torch.mv(A, rk) - b
#            print('hrg',Hrg.norm())
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gama = 0  
            phi = phil
            rk = rkl
            x = xl
            flag = 2
            print('flag = 2, b is not in the range(A)!')
            maxit += 1
            
        ## stopping criteria
        xnorm = torch.norm(x)   
        rnorm = phi
        relres = rnorm / beta1
        
        if phi < rtol*xnorm:
            if torch.dot(x, b) >= psi*xnorm**2:
                flag = 1  ## trustful least square solution
                return x, relres, iters, rk, dType
        if iters > maxit:
            flag = 4  ## exit before maxit
            print('Maximun iteration reached', flag, iters)
            return x, relres, iters, rk, dType
    return x, relres, iters, rk, dType

# =============================================================================
def myMINRES_TR(A, b, rtol, maxit, Delta=1, shift=0, psi=1E-6,
               reOrth=True, isZero=1E-10):
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
    Flag = -1, b and x are eigenvectors.
    Flag = 0, the exact solution is x = 0.
    Flag = 1, min-length solution for the LS problem, given rtol.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    
    If Flag = 0, then rank(A) = iters - 1.
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
    dltan = 0
    eplnn = 0
    gama = 0
    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    dType = 'Sol'
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b)     
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    # if b.T.dot(Ax(A, b)) <= 0: # covered
    #     flag = 3
    #     #print('ncb', -nc)
    #     return b, relres, iters
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
    while flag == flag0:
        #lanczos
        betal = beta
        beta = betan
        v = vn
        r3 = Ax(A, v)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
#        if noprecon:
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
                    print('flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
                    print('flag = -1')
                    break
            
        vn = r3/betan
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)  
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
        
        nc = csl * gbar
        # #print('nc', nc)
        if nc >= -isZero:
            flag = 3
            #print('nc', -nc)
            dType = 'NC'
            if iters == 1:                
                m_r = - phil*Delta - Delta**2*nc/2
                return Delta*rkl/phil, relres, iters, m_r, dType, Delta
            else:
                m_r = - phil*Delta - Delta**2*nc/2
                d, m_p, iters = solvetm(A, b, xl, rkl, Delta, xnorm, phil, m_r, iters)
                relres += 1
                return d, relres, iters, m_p, dType, Delta
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            rk = sn**2 * rkl - phi * cs * vn
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gama = 0  
            phi = phil
            rk = rkl
            x = xl
            flag = 2
            print('flag = 2')
            maxit += 1
            
        ## stopping criteria
        xnorml = xnorm
        xnorm = torch.norm(x)    
        # print(iters, xnorm)
        rnorm = phi
        relres = rnorm / beta1
        # print(iters, relres)
        # m0 = -x.T.dot(b) + x.T.dot(Ax(A,x))/2
        
        if xnorm > Delta:  
            flag = 5
            tw = tau*w
            twnorm = torch.norm(tw)
            xlTtwl = torch.dot(xl, tw)
#                xTr = x.T.dot(rk)
#                print(xlTrl, xTr)
            omega = (-xlTtwl + torch.sqrt(xlTtwl**2 + twnorm**2*(
                Delta**2 - xnorml**2)))/twnorm**2
            x = xl + omega*tw    
            if torch.dot(x, b) >= psi*xnorm**2:
                mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
                relres += 1
                return x, relres, iters, mp, dType, Delta
                
        if phi < rtol*xnorm:
            if torch.dot(x, b) >= psi*xnorm**2:
                flag = 1  ## trustful least square solution
                mp = torch.dot(x, -b) + torch.dot(x, b - rk)/2
                return x, relres, iters, mp, dType, Delta
        
        if iters > maxit:
            flag = 4  ## exit before maxit
            print('Maximun iteration reached', flag, iters)
    mp = -torch.dot(x, rk)/2 - torch.dot(x, b)/2
    # if -mp < 1e-19:
    #     print(' ')
    return x, relres, iters, mp, dType, Delta

def myMINRES_TR_NS(A, b, rtol, maxit, Delta=1, shift=0, psi=1E-6,
                   reOrth=True, isZero=1E-10):
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
    Flag = -1, b and x are eigenvectors.
    Flag = 0, the exact solution is x = 0.
    Flag = 1, min-length solution for the LS problem, given rtol.
    Flag = 2, ||b - Ax|| system inconsistent, given rtol cannot reach, 
        return the best Min-length solution x.
    flag = 3, non-positive curvature direction detected.
    Flag = 4, the iteration limit was reached.
    
    If Flag = 0, then rank(A) = iters - 1.
    """        
    
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
    
    # rtol = np.sqrt(rtol)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    dltan = 0
    eplnn = 0
    gama = 0
    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    dType = 'Sol'
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b)      
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
    no_stop = False
    m_iters = -1
    while flag == flag0 or no_stop:
        #lanczos
        betal = beta
        beta = betan
        v = vn
        r3 = Ax(A, v)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
#        if noprecon:
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
                    print('flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
                    print('flag = -1')
                    break
            
        vn = r3/betan
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
        
        nc = csl * gbar
        # #print('nc', nc)
        if nc >= -isZero:
            flag = 3
            if no_stop == False:
                m_iters = 0
            no_stop = True
            dType = 'NC'            
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            rk = sn**2 * rkl - phi * cs * vn
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gama = 0  
            phi = phil
            rk = rkl
            x = xl
            flag = 2
            print('flag = 2')
            maxit += 1
            
        if no_stop:
            if m_iters == 0:
                m_rl = - phil*Delta - Delta**2*nc/2
#                print('m_rl_best=', m_rl)
                m_bestl = m_rl
                p_bestl = Delta*rkl/phil
                m_best = m_bestl
                p_best = p_bestl
            if m_iters > 0:
                m_bestl = m_best
                p_bestl = p_best
                m_rl = - phil*Delta - Delta**2*nc/2  
                if m_rl < m_best:
                    m_best = m_rl
#                    print('m_rl', m_best)
                    p_best = Delta*rk/phi
            # lambda x      
            if m_iters == 0:      
                m_xl = torch.dot(xl, -b) + torch.dot(xl, b-rkl)/2
                if m_xl < m_best:
                    m_best = m_xl
#                    print('m_xl', m_best)
                    p_best = xl
                
            xTr = torch.dot(xl, rkl)
            xnorm = xl.norm()
            diff = Delta**2 - xnorm**2
            omega = (-xTr + torch.sqrt(xTr**2 + phil**2*diff))/phil**2
            z = xl + omega*rkl   
            zTAz = torch.dot(xl, b - rkl) - omega**2 * csl * gbar*phil**2 # xlTArkl = 0
            m_xl_rl = torch.dot(z, -b) + zTAz/2
            if m_xl_rl < m_best:
                m_best = m_xl_rl
                p_best = z     
                
            if x.norm() > Delta:
                return p_best, relres, iters, m_best, dType, Delta
                
            m_x = torch.dot(x, -b) + torch.dot(x, b-rk)/2      
            if m_x < m_best:
                m_best = m_x
                p_best = x      
            
            xTr = torch.dot(x, rkl)
            xnorm = x.norm()
            diff = Delta**2 - xnorm**2
            omega = (-xTr + torch.sqrt(xTr**2 + phil**2*diff))/phil**2
            z = x + omega*rkl   
            zTAz = torch.dot(x + 2*omega*rkl, b - rk) - omega**2 * csl * gbar*phil**2
            # print(zTAz, torch.dot(z, Ax(A, z)))
            m_x_rl = torch.dot(z, -b) + zTAz/2
            if m_x_rl < m_best:
                m_best = m_x_rl
#                print('m_x_rl', m_best)
                p_best = z     
            
            m_iters += 1
#            print('m_rl', m_rl, m_xl, m_x, m_xl_rl, m_x_rl) 
            
            if torch.isnan(m_best):
                return p_bestl, relres, iters, m_bestl, dType, Delta
                
            if m_best == m_bestl:
#                print('m_best_return', m_best, m_iters)
                return p_best, relres, iters, m_best, dType, Delta
        ## stopping criteria
        xnorml = xnorm
        xnorm = torch.norm(x)    
        rnorm = phi
        relres = rnorm / beta1
        
        if not no_stop:            
            if xnorm > Delta:  
                flag = 5
                tw = tau*w
                twnorm = torch.norm(tw)
                xlTtwl = torch.dot(xl, tw)
    #                xTr = x.T.dot(rk)
    #                print(xlTrl, xTr)
                omega = (-xlTtwl + torch.sqrt(xlTtwl**2 + twnorm**2*(
                    Delta**2 - xnorml**2)))/twnorm**2
                x = xl + omega*tw    
                if torch.dot(x, b) >= psi*xnorm**2:
                    mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
                    relres += 1
                    return x, relres, iters, mp, dType, Delta
            
            if relres <= rtol:
                flag = 1  ## exit before maxit            
                if torch.dot(x, b) >= psi*xnorm**2:
                    flag = 1  ## trustful least square solution
                    mp = torch.dot(x, -b) + torch.dot(x, b-rk)/2
                    return x, relres, iters, mp, dType, Delta
                # print('Sol')
            if iters > maxit:
                flag = 4  ## exit before maxit
                print('Maximun iteration reached', flag, iters)
    mp = -torch.dot(x, rk)/2 - torch.dot(x, b)/2
    # if -mp < 1e-19:
    #     print(' ')
    return x, relres, iters, mp, dType, Delta
# =============================================================================

# =============================================================================
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




if __name__ == '__main__':
    main()
    
    
    
    
