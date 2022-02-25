# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:20:01 2021

@author: Liu Yang
"""
import torch
def myGMRES(A, b, rtol, maxit, shift=0, reOrth=True, isZero=1E-15):
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
    
    x = torch.zeros_like(b)
    H = torch.zeros(2,1,device=b.device, dtype=b.dtype) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
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
        
        iters += 1
        if iters == 1:
            if betan < isZero:
                break
                
            
        vn = r3/betan
        
                
        Q, R = house(H)
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
        Hn = torch.zeros(iters+2, iters+1,device=b.device, dtype=b.dtype)
        Hn[:iters+1,:iters] = H
        
        H = Hn
        Ue = Q[0,:-1]
        x = GMRES_return(R, Ue, beta1, iters, Vn)
        
        relres2 = 1 - Ue.norm()**2
        relres = torch.sqrt(relres2)
            
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
    m, n = A.shape
    R = torch.zeros(m, n,device=A.device, dtype=A.dtype)
    Q = torch.eye(m,device=A.device, dtype=A.dtype)
    R = torch.clone(A)
    for i in range(n):
        a1 = R[i:m,i]
        e1 = torch.zeros(m-i,device=A.device, dtype=A.dtype)
        e1[0] = 1
        u = a1 + torch.sign(a1[0])*a1.norm()*e1
        u = u/u.norm()
        R[i:m, i:n] = R[i:m, i:n] - torch.ger(2*u, torch.mv(R[i:m, i:n].T, u))
        P = torch.eye(m, dtype=A.dtype)
        P[i:m, i:m] = P[i:m, i:m] - torch.ger(2*u, u)
        Q = torch.mm(P, Q)
    return Q.T, R


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = torch.mv(A, x)
    return Ax

def main(): 
    n = 30
    b = torch.randn(n, dtype=torch.float64)
    A = torch.randn(n, n, dtype=torch.float64)
    rtol = 0
    isZero=1E-8
    x1, relres1, iters1 = myGMRES(A, b, rtol, n, isZero=isZero)
    print(torch.inverse(A) @ b - x1)

if __name__ == '__main__':
    main()
    