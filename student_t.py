# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:12:37 2021

@author: Liu Yang
"""

import torch
from derivativetest import derivativetest
import numpy as np
from regularizer import regConvex

# def student_t2(A, b, x, nu=1, HProp=1, arg=None, reg=None):
#     """
#     returns  f, g, Hv of sum(log(1+||Ax-b||^2/nu))/n
#     """
#     if reg == None:
#         reg_f = 0
#         reg_g = 0
#         reg_Hv = lambda v: 0
#     else:
#         reg_f, reg_g, reg_Hv = reg(x)
        
#     r = torch.mv(A, x) - b
#     n = A.shape[0]
#     a = r/nu
#     b = 1 + r*a
#     c = b*nu
#     f = sum(torch.log(b))/n + reg_f
#     if arg == 'f':
#         return f.detach()
    
#     g = torch.mv(A.T, 2*a/b)/n + reg_g
#     if arg == 'g':
#         return g.detach()
#     if arg == 'fg':
#         return f.detach(), g.detach()  
        
    
#     if arg is None:
#         if HProp == 1:
#             s = (2*b/nu - 4*a**2)/b**2/n
#             Hv = lambda v: (torch.mv(A.T, s*torch.mv(A, v)) + reg_Hv(v)).detach()
#             return f.detach(), g.detach(), Hv 
#         else:
#             n_H = np.int(np.floor(n*HProp))
#             idx_H = np.random.choice(n, n_H, replace = False)    
#             s = (2*b/nu - 4*a**2)/b**2/n_H
#             Hv = lambda v: (torch.mv(A[idx_H,:].T, s[idx_H]*torch.mv(
#                     A[idx_H,:], v)) + reg_Hv(v)).detach()
#             return f.detach(), g.detach(), Hv  

def student_t(A, b, x, nu=1, HProp=1, arg=None, reg=None,
                 index=None):
    """
    returns  f, g, Hv of sum(log(1+||Ax-b||^2/nu))/n
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(x)
        
    if index is not None:
        A = A[:,index]
    r = torch.mv(A, x) - b
    n = A.shape[0]
    a = r/nu
    b = 1 + r*a
    f = sum(torch.log(b))/n + reg_f
    if arg == 'f':
        return f.detach()
    
    g = torch.mv(A.T, 2*a/b)/n + reg_g
    if arg == 'g':
        return g.detach()
    if arg == 'fg':
        return f.detach(), g.detach()  
        
    
    if arg is None:
        if HProp == 1:
            s = (2*b/nu - 4*a**2)/b**2/n
            Hv = lambda v: (torch.mv(A.T, s*torch.mv(A, v)) + reg_Hv(v)).detach()
            return f.detach(), g.detach(), Hv 
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)    
            s = (2*b/nu - 4*a**2)/b**2/n_H
            Hv = lambda v: (torch.mv(A[idx_H,:].T, s[idx_H]*torch.mv(
                    A[idx_H,:], v)) + reg_Hv(v)).detach()
            return f.detach(), g.detach(), Hv  

def reg_student_t(x, nu=1, arg=None):
    """
    returns  f, g, Hv of log(1+||x||^2/nu)
    """
    xn2 = torch.dot(x, x)
    xnu = 1+xn2/nu
    f = torch.log(xnu)
    if arg == 'f':
        return f.detach()
    
    g = 2/nu/xnu*x
    if arg == 'g':
        return g.detach()
    if arg == 'fg':
        return f.detach(), g.detach()  
    
    Hv = lambda v: ((2*xnu/nu*v - 4*torch.dot(x,v)*x/nu/nu)/xnu**2).detach()
    
    if arg is None:
        return f.detach(), g.detach(), Hv   
#@profile
def main():        
    import torch.utils.data as data
    import torchvision.datasets as datasets
    from  torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    data_dir = '../Data'
    train_Set = datasets.MNIST(data_dir, train=True,
                                transform=transforms.ToTensor())
    (n, d, d) = train_Set.data.shape
    d = d**2
    X = train_Set.data.reshape(n, d)
    X = X[:1000,:]/255
    X = X.double()
    Y_index = train_Set.targets
    Y = (Y_index>5)*torch.tensor(1).double()
    Y = Y[:1000]
    
#    print(Y)
    lamda = 0
    w = torch.randn(d, dtype=torch.float64)
#    reg = None
    reg = lambda x: regConvex(x, lamda)
    # reg = lambda x: regNonconvex(x, lamda)
    fun1 = lambda x, arg=None: student_t(X, Y, x, arg=arg, reg=reg)
#    fun1 = lambda x, arg=None: reg_student_t(x, arg=arg)
    derivativetest(fun1,w)    
#    
if __name__ == '__main__':
    main()