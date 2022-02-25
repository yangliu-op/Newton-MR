# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:20:01 2021

@author: Liu Yang
"""
import torch
def myGMRES(A, b, rtol, maxit, shift=0, reOrth=True, isZero=1E-15):
    
    n = len(b)
    device = b.device
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
    
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    vn = r3/betan
    
    x = torch.zeros(n,device=device, dtype=b.dtype)
    H = torch.zeros(2,1,device=device, dtype=b.dtype) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        Rt = b.reshape(-1, 1)
#    normHy2 = tau**2
    while flag == flag0:
        #Arnoldi
        v = vn
        r3 = Ax(A, v)
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        for i in range(iters+1):
            h_ij = torch.dot(r3, Vn[:,i])
            r3 = r3 - h_ij*Vn[:,i]
            H[i,iters] = h_ij
        
#        if reOrth:
#            print(torch.mv(Vn, torch.mv(Vn.T, r3)).norm())
#            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
             
        betan = torch.norm(r3)
        H[iters+1,iters] = betan
#        print(betan)

        
        iters += 1
        if iters == 1:
            if betan < isZero:
                break
                
            
        vn = r3/betan
        
                
        Q, R = house(H)
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
            rt = Q[:,-1][0]*beta1 * Vn @ (Q[:,-1])
            Rt = torch.cat((Rt, rt.reshape(-1, 1)), axis=1)
        Hn = torch.zeros(iters+2, iters+1,device=device, dtype=b.dtype)
        Hn[:iters+1,:iters] = H
#        print(Hn)
        H = Hn
        Ue = Q[0,:-1]
        x = GMRES_return(R, Ue, beta1, iters, Vn)
        print((b - A @ x).norm()/b.norm(), abs(Q[:,-1][0]))
        print('r', Rt.T @ (A @ Rt))
        
        relres2 = 1 - Ue.norm()**2
        if relres2 < 0:
            relres = -relres2*0
        else:
            relres = torch.sqrt(relres2)
#        print(iters, relres)
        # m0 = -x.T.dot(b) + x.T.dot(Ax(A,x))/2
        if relres <= rtol:
            return x, relres, iters
#        print(iters, maxit)
        if iters >= maxit:
            flag = 4  ## exit before maxit
            x = GMRES_return(R, Ue, beta1, iters, Vn)
            return x, relres, iters
    x = GMRES_return(R, Ue, beta1, iters, Vn)
    return x, relres, iters

def GMRES_return(R, Ue, beta1, iters, Vn):
    if iters == 1:
        y = beta1*Ue/R[:-1,:]
        x = Vn[:,:-1][:,0] * y[0]
    else:
        y = beta1*backward_sub(R[:-1,:], Ue)
        x = Vn[:,:-1] @ y
    return x

def backward_sub(A, b):
    # A is upper triangular
    n = len(b)
    x = torch.zeros_like(b)
    for k in range(n):
        x[n-k-1] = b[n-k-1]
        for j in range(k):
            x[n-k-1] -= A[n-k-1,n-j-1]*x[n-j-1]
        x[n-k-1] = x[n-k-1]/A[n-k-1,n-k-1]
    return x

def house(A):
    device = A.device
    m, n = A.shape
    R = torch.zeros(m, n,device=device, dtype=A.dtype)
    Q = torch.eye(m,device=device, dtype=A.dtype)
    R = torch.clone(A)
    for i in range(n):
        a1 = R[i:m,i]
        e1 = torch.zeros(m-i,device=device, dtype=A.dtype)
        e1[0] = 1
        u = a1 + torch.sign(a1[0])*a1.norm()*e1
        u = u/u.norm()
        R[i:m, i:n] = R[i:m, i:n] - torch.ger(2*u, torch.mv(R[i:m, i:n].T, u))
        P = torch.eye(m, dtype=A.dtype)
        P[i:m, i:m] = P[i:m, i:m] - torch.ger(2*u, u)
        Q = torch.mm(P, Q)
    return Q.T, R

def myMINRES_Arnoldi(A, b, rtol, maxit, shift=0, reOrth=True, isZero=1E-15):
    device = b.device
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
    
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
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
    H = torch.zeros(2,1,device=device, dtype=b.dtype) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    # if b.T.dot(Ax(A, b)) <= 0: # covered
    #     flag = 3
    #     #print('ncb', -nc)
    #     return b, relres, iters
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
#    normHy2 = tau**2
    while flag == flag0:
        #lanczos
#        betal = beta
#        beta = betan
        v = vn
#        print(v)
        r3 = Ax(A, v)
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        for i in range(iters+1):
            h_ij = torch.dot(r3, Vn[:,i])
            r3 = r3 - h_ij*Vn[:,i]
            H[i,iters] = h_ij
        alfa = H[iters,iters]
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        
        
        
#        alfa = torch.dot(r3, v)
#        if iters > 1:
#            r3 = r3 - r1*beta/betal            
#        r3 = r3 - r2*alfa/beta
#        
#        if reOrth:
#            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))        
        
        betan = torch.norm(r3)
        H[iters+1,iters] = betan

        
        iters += 1
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
        
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
            if iters == 1:
                alpha_diag = alfa.reshape(1)
                beta_diag = betan.reshape(1) 
                T = alfa
#                print((torch.mv(A, v) - alfa*v).norm(), betan)
                bT = torch.cat((T.reshape(1), betan.reshape(1)), 0).reshape(2, 1)
            else:
                alpha_diag = torch.cat((alpha_diag, alfa.reshape(1)), axis=0)
                beta_diag = torch.cat((beta_diag, betan.reshape(1)), axis=0)   
                T = torch.diag(beta_diag[:-1],-1) + torch.diag(
                        alpha_diag,0) + torch.diag(beta_diag[:-1],1)  
#                print('H', H[:-1,:])
#                print('T', T)
#                print('H-T',(H[:-1,:] - T).norm())
                
        Hn = torch.zeros(iters+2, iters+1,device=device, dtype=b.dtype)
        Hn[:iters+1,:iters] = H
#        print(Hn)
        H = Hn
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
        
#        nc = csl * gbar 
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
        print(iters, gama)
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            
#            normHy2 = tau**2 + normHy2
#            print((beta1**2 - phi**2)/x.norm()**2, end=' ')
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
#            print('inner_b is not in the range(A)!', flag, iters, relres)
            maxit += 1
            
        # print('rk', iters, phi/beta1)
        ## stopping criteria
        xnorm = torch.norm(x)   
        print(iters, xnorm)
        rnorm = phi
        relres = rnorm / beta1
#        print(iters, relres)
        # m0 = -x.T.dot(b) + x.T.dot(Ax(A,x))/2
        if relres <= rtol:
            return x, relres, iters, LSType, dType
#        print(iters, maxit)
        if iters > maxit:
            flag = 4  ## exit before maxit
#            print('inner_Maximun iteration reached', flag, iters, relres)
            return x, relres, iters, LSType, dType
    return x, relres, iters, LSType, dType

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


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = torch.mv(A, x)
    return Ax

def main(): 
    """Run an example of minresQLP."""
#    torch.manual_seed(2000888)
#    from myMINRES import myMINRES_naive
#    n = 3
#    b = torch.randn(n, dtype=torch.float64)
#    A = torch.randn(n, n, dtype=torch.float64)
##    A[1, 1] = 2
##    A[2, 2] = 3
#    B = A + A.T
#    U, s, V = torch.svd(B)
#    ss = abs(s)
##    ss[0] = 1E-3
#    print(min(ss))
##    ss[-1] = 0 
#    E = U @ torch.diag(ss) @ U.T
##    print(E)
#    rtol = 0
#    C = torch.randn(n, n, dtype=torch.float64)*1E-4
##    D = E+C
#    D = E
#    isZero=1E-8
#    L = max(ss)
##    x, relres, iters, LSType, dType = myMINRES_naive(E, b, rtol, n, isZero=isZero, L=L)
##    print(x.norm())
##    print(torch.inverse(A) @ b)
#    x1, relres1, iters1 = myGMRES(A, b, rtol, n, isZero=isZero)
#    print(torch.inverse(A) @ b - x1)
#    print('x-x1', (x-x1).norm()/x1.norm(), x1.norm(),relres1, iters1, x - x1)
#    
#    x2, relres2, iters2, LSType2, dType2 = myMINRES_Arnoldi(D, b, rtol, n, isZero=isZero)
#    print('x-x2', (x-x2).norm()/x2.norm(),x2.norm(), relres2)
#    x2, relres2, iters2, LSType2, dType2 = myMINRES_Arnoldi(D, b, rtol, n)
#    print(x1 - torch.inverse(D)@b)
# =============================================================================
#   Pseudo-inverses Rank1 no errors, why error large?
    torch.manual_seed(7)
    
    from myMINRES import myMINRES_naive, mrIIDet
    import scipy as sp
    from MinresQLP import MinresQLP
    n = 23
    b = torch.randn(n, dtype=torch.float64)
    A = torch.randn(n, n, dtype=torch.float64)
    B = A + A.T
#    A = torch.diag(torch.ones(n, dtype=torch.float64))
#    A[0,1] = 2
    B = A + A.T
    U, s, V = torch.svd(B)
#    ss = abs(0,3,n)
    ss = torch.logspace(0,1,n, dtype=torch.float64)
    ss = ss
#    ss[0] = 1E-2
#    ss[2] = 1E3
    d = 20
    ss[d:] = 0
    print(ss)
    E = U @ torch.diag(ss) @ U.T
    U1 = U[:,:d]
    U2 = U[:,d:]
#    print(U1.shape)
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    b1 = P1 @ b
    b2 = P2 @ b
#    print(b, b1 + b2)
#    print(b1.norm()/b.norm())
    b1 = b1 * 1E0
    b = b1 + b2
    maxit = d
#    x1, relres1, iters1 = myGMRES(E, b, 0, n)
    x, relres, iters, rk, phi = myMINRES_naive(E, b, 0,  maxit, isZero=1E-10)
#    x3, relres3, iters3, rk3, phi3 = mrIIDet(E, b, 0,  maxit, isZero=1E-10, detector=True)
#    x,relres,iters = MinresQLP(E, b, 0, n)
#    print('UUb', b1.norm()**2/b.norm()**2)
#    print('UUz', (P1 @ x).norm()**2/x.norm()**2, (max(ss)/min(ss[ss>0]))**2*(P1 @ x).norm()**2/x.norm()**2)
#    print('relres',iters,phi, (E @ rk).norm())
    Ed = torch.pinverse(E)
    nu = (P1 @ b).norm()**2/b.norm()**2
    print('nu', nu)
#    print(Ed)
    xd = Ed@b
    Eds = sp.linalg.pinv(E)
#    print(Eds)
    xds = Eds.dot(b.numpy())
    xd1 = x - x@rk/rk.norm()**2*(rk)
    print(x.norm(), xd.norm(), xd1.norm(), (xd1 - (P1 @ xd1)).norm())
#    print('x', x)
#    print('xd1', xd1)
##    print('MR2', x3)
#    print('pseudo', xd)
##    print('scipy.pinv', xds)
##    print('torch.pinv', xd)
#    print((P2 @ xd1).norm(), (P2 @ x).norm())
#    print((P1 @ x).norm(), (P1 @ xd1).norm())
#    print((xd1 @ (E @ xd1)).norm(), (x @ (E @ xd1)).norm(), (x @ (E @ x)).norm())
#    print((x @ b).norm(), (xd1 @ b).norm())
#    print('Range', xd1 - (P1 @ xd1))
#    print('Null', x - xd1, P2 @ x)
#    print('Ax', E @ x, E @ xd1)
#    print('11', xd1 @ ( P1 @ b), xd1 @ (E @ x ), xd1 @ b, x @ (E @ xd1 ), x @ (E @ x ), xd1 @ rk)
#    print(' ', (x @ rk)**2/x.norm()**2)
# =============================================================================
    
#    torch.manual_seed(2)
#    
#    from myMINRES import myMINRES_naive, mrIIDet
#    import scipy as sp
#    from MinresQLP import MinresQLP
#    n = 23
#    b = torch.randn(n, dtype=torch.float64)
#    A = torch.randn(n, n, dtype=torch.float64)
#    R = torch.randn(n, n, dtype=torch.float64)
#    B = A + A.T
##    A = torch.diag(torch.ones(n, dtype=torch.float64))
##    A[0,1] = 2
#    B = A + A.T
#    U, s, V = torch.svd(B)
#    ss = abs(s)
#    ss = ss
##    ss[0] = 1E-1
##    ss[2] = 1E3
#    d = 20
#    ss[d:] = 0
#    print(ss)
#    E = U @ torch.diag(ss) @ U.T
#    U1 = U[:,:d]
#    U2 = U[:,d:]
##    print(U1.shape)
#    P1 = U1 @ U1.T
#    P2 = U2 @ U2.T
#    b1 = P1 @ b
#    b2 = P2 @ b
#    
#    U5, s5, V5 = torch.svd(R + R.T)
##    s5[d-3:] = 0
#    U6 = U5[:,:d]
#    P5 = U6 @ U6.T
#    P6 = U5[:,d:] @ U5[:,d:].T 
#    F = U @ torch.diag(s5) @ U.T
##    print(b, b1 + b2)
##    print(b1.norm()/b.norm())
#    b1 = b1 * 1E-1
#    b = b1 + b2
#    maxit = 20
#    x, relres, iters, rk, phi = myMINRES_naive(E, b, 0,  maxit, isZero=1E-10)
##    x3, relres3, iters3, rk3, phi3 = mrIIDet(E, b, 0,  maxit, isZero=1E-10, detector=True)
##    x,relres,iters = MinresQLP(E, b, 0, n)
#    print('UUb', b1.norm()**2/b.norm()**2)
#    print('UUz', (P1 @ x).norm()**2/x.norm()**2, (max(ss)/min(ss[ss>0]))**2*(P1 @ x).norm()**2/x.norm()**2)
#    print(iters,phi)
#    Ed = torch.pinverse(E)
#    Fd = torch.pinverse(F)
#    EFd = torch.pinverse(E @ F)
##    x5d = Fd@b
#    nu = (P1 @ b).norm()**2/b.norm()**2
#    print('nu', nu)
##    print(Ed)
#    xd = Ed@b
#    Eds = sp.linalg.pinv(E)
##    print(Eds)
#    xds = Eds.dot(b.numpy())
#    xd1 = x - x@rk/rk.norm()**2*(rk)
#    print(x.norm(), xd.norm(), xd1.norm())

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    