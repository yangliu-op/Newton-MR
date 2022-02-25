# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:27:46 2020

@author: Yang Liu
"""
import torch
from derivativetest import derivativetest, derivativetest_class

class logitreg_class(object):
    """
    Logistic regression
    INPUT:
        A: Matrix
        x: variable vector
        b: label
        arg: output control
    OUTPUT:
        f, gradient, Hessian-vector product
    """
    def __init__(self, A, b, x, HProp=None):
        self.A = A
        self.n, self.d = A.shape
        self.b = b
        self.x = x
        self._t = torch.mv(self.A, self.x)
        self._M = torch.clamp(self._t, min=0)
        
    def _sum_exp(self):
        return torch.exp(-self._M) + torch.exp(self._t - self._M)
        
    def f(self):
        # x = x.reshape(d,1) 
        fx = self._M + torch.log(self._sum_exp()) - self.b*self._t
        fx = sum(fx)/self.n
        return fx
        
    def frac(self):        
        return torch.exp(self._t - self._M)/self._sum_exp()
    
    def g(self):
        return torch.mv(self.A.T, self.frac() - self.b)/self.n
    
    def Hv(self, v):
        D = self.frac()*(1-self.frac())
        return torch.mv(self.A.T, D*torch.mv(self.A, v))/self.n

def logitreg(A, b, x, HProp=None, arg=None, reg=None):
    """
    Logistic regression
    INPUT:
        A: Matrix
        x: variable vector
        b: label
        arg: output control
    OUTPUT:
        f, gradient, Hessian-vector product
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(x)
    #s = 1/(1+ np.exp(-t)) =  e^t/(1+e^t) = e^(t-M)/a
    #a = e^(-M)+e^(t-M)
    n, d = A.shape
    # x = x.reshape(d,1) 
    t = torch.mv(A, x)
    M = torch.clamp(t, min=0)
    # power = torch.cat((-M, t-M), axis = 1)
    a = torch.exp(-M) + torch.exp(t-M)
    f = M + torch.log(a) - b*t
    f = sum(f)/n + reg_f
#    print('f', f)
    if arg == 'f':        
        return f
    
    s = torch.exp(t-M)/a
    g = torch.mv(A.T, s - b)/n + reg_g
    
    if arg == 'g':        
        return g
    if arg == 'fg':        
        return f, g
    
    if arg == None:
        D = s*(1-s)
        Hv = lambda v: torch.mv(A.T, D*torch.mv(A, v))/n + reg_Hv(v)
        return f, g, Hv
    

def logitreg2(A, b, x, arg=None):
    """
    Logistic regression
    INPUT:
        X: Matrix
        w: variable vector
        arg: output control
    OUTPUT:
        f, gradient, Hessian
    """
    #s = 1/(1+ np.exp(-t)) =  e^t/(1+e^t) = e^(t-M)/a
    #a = e^(-M)+e^(t-M)
    n, d = A.shape
    f = 0
    g = torch.zeros_like(b)
    Hv = torch.zeros(d,d, device = b.device, dtype=b.dtype)
    for i in range(n):
        bi = b[i]
        ai = A[i]
        t = torch.dot(ai, x)
        M = torch.clamp(t, min=0)
        a = torch.exp(-M) + torch.exp(t-M)
        f += torch.log(a/torch.exp(-M)) - bi*t        
        s = torch.exp(t-M)/a
        g += ai*(s - bi)        
        # Hv += ai.T * s*(1-s) * ai
        Hv += s*(1-s) *torch.outer(ai, ai)
        # Hv = lambda v: A.T.dot(D*(A.dot(v)))
    return f, g, Hv
    
def main():
    # rand.seed(83)
    n = 100
    d = 50
    A = torch.randn(n,d, dtype=torch.float64)
    b = torch.ones(n, dtype=torch.float64)
    w = torch.randn(d, dtype=torch.float64)
    fun = lambda x, arg=None: logitreg(A,b,x, arg=arg)
    derivativetest(fun,w)    
#    fun = lambda x: logitreg2(A,b,x)
    
#    fun = lambda x: logitreg_class(A,b,x)
#    derivativetest_class(fun,w)   
    
#    derivativetest(fun1,w)    
if __name__ == '__main__':
    main()