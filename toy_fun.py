# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:10:08 2021

@author: Liu Yang
"""
import torch
from derivativetest import derivativetest, derivativetest_class

def saddle(x, arg=None):
    x1 = x[0]
    x2 = x[1]
    f = x1**4 + 2*(x1 - x2)*x1**2 + 4*x2**2
    if arg == 'f':
        return f
    g = torch.tensor([4*x1**3 + 6*x1**2 - 4*x1*x2, -2*x1**2 + 8*x2])
    if arg == 'g':
        return g
    if arg == 'fg':
        return f, g
    H = torch.tensor([[12*x1**2 + 12*x1 - 4*x2, -4*x1], [-4*x1, 8]])
    if arg == None:
        return f, g, H

def log_fun(x, c=0.1, arg=None):
    s = torch.dot(x, x) + c
    f = torch.log(s)
    if arg == 'f':
        return f.detach()
    d = 2/s
    g = d*x
    if arg == 'g':
        return g.detach()
    if arg == 'fg':
        return f.detach(), g.detach()
    def Hvp(v):
        xTv = torch.dot(x, v)
        Hv = d*(v - d*xTv*x)
        return Hv.detach()
    Hv = lambda v: Hvp(v)
    if arg == None:
        return f.detach(), g.detach(), Hv
    
        
class log_fun_class(object):
    def __init__(self, x, c=0.1):
        self.x = x
        self.c = c
        self.inlog = torch.dot(self.x, self.x) + self.c
        self.tmp = 2/self.inlog

    def f(self):
        fx = torch.log(self.inlog)
        return fx.detach()
    
    def g(self):
        gx = self.tmp*self.x
        return gx.detach()
        
    def Hv(self, v):
        xTv = torch.dot(self.x, v)
        Hv = self.tmp*(v - self.tmp*xTv*self.x)
        return Hv.detach()
    
def main():     
    d = 2
    w = torch.ones(d, dtype=torch.float64)
    # reg = None
#    fun = lambda x: log_fun_class(x)
#    derivativetest_class(fun,w)  
#    
#    fun = lambda x, arg=None: log_fun(x, arg=arg)
    fun = lambda x, arg=None: saddle(x, arg=arg)
    derivativetest(fun,w)    
#    print(' ')

if __name__ == '__main__':
    main()