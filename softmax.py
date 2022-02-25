import numpy as np
# from scipy.sparse import spdiags, identity
import numpy.random as rand
# from numpy import matlib as mb
from derivativetest import derivativetest, derivativetest_class
# from scipy.linalg import block_diag
from regularizer import regConvex, regNonconvex
import torch

def softmax(X, Y, w, HProp = 1, arg=None, reg=None):  
    """
    All vectors are column vectors.
    INPUTS:
        X: a nxd data matrix.
        Y: a nxC label matrix, C = total class number - 1
        w: starting point
        HProp: porposion of Hessian perturbation
        arg: output control
        reg: a function handle of regulizer function that returns f,g,Hv
    OUTPUTS:
        f: objective function value
        g: gradient of objective function
        Hv: a Hessian-vector product function handle of any column vector v
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(w)
    global d, C
    n, d = X.shape
    C = int(len(w)/d)
    w = w.reshape(d*C,1) #[d*C x 1]
    W = w.reshape(C,d).T #[d x C]
    XW = torch.mm(X,W) #[n x C]
    large_vals = torch.max(XW,axis = 1)[0] #[n,1 ]
    large_vals = torch.clamp(large_vals, min=0) #M(x), [n, 1]
    #XW - M(x)/<Xi,Wc> - M(x), [n x C]
    XW_trick = XW - large_vals.repeat(C, 1).T
    #sum over b to calc alphax, [n x total_C]
    XW_1_trick = torch.cat((-large_vals.reshape(-1,1), XW_trick), 1)
    #alphax, [n, ]
    sum_exp_trick = torch.sum(torch.exp(XW_1_trick), axis = 1)
    log_sum_exp_trick = large_vals + torch.log(sum_exp_trick)  #[n, 1]
    
    f = torch.sum(log_sum_exp_trick)/n - torch.sum(torch.sum(XW*Y,axis=1))/n + reg_f
    if arg == 'f':        
        return f
    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = inv_sum_exp.repeat(C, 1).T
    S = inv_sum_exp*torch.exp(XW_trick) #h(x,w), [n x C] 
    g = torch.mm(X.T, S-Y)/n #[d x C]
    g = g.T.flatten() + reg_g#[d*C, ]  
    
    if arg == 'g':
        return g    
    
    if arg == 'fg':
        return f, g

    if HProp == 1:
        Hv = lambda v: hessvec(X, S, n, v) + reg_Hv(v) #write in one function to ensure no array inputs        
        return f, g, Hv
    else:
        n_H = np.int(np.floor(n*HProp))
        idx_H = np.random.choice(n, n_H, replace = False)
        Hv = lambda v: hessvec(X[idx_H,:], S[idx_H,:], n_H, v) + reg_Hv(v)
        return f, g, Hv

def hessvec(X, S, n, v):
    v = v.reshape(len(v),1)
    V = v.reshape(C, d).T #[d x C]
    A = torch.mm(X,V) #[n x C]
    AS = torch.sum(A*S, axis=1)
    rep = AS.repeat(C, 1).T #A.dot(B)*e*e.T
    XVd1W = A*S - S*rep #[n x C]
    Hv = torch.mm(X.T, XVd1W)/n #[d x C]
    Hv = Hv.T.flatten() #[d*C, ] #[d*C, ]
    return Hv


class softmax_class(object):  
    """
    All vectors are column vectors.
    INPUTS:
        X: a nxd data matrix.
        Y: a nxC label matrix, C = total class number - 1
        w: starting point
        HProp: porposion of Hessian perturbation
        arg: output control
        reg: a function handle of regulizer function that returns f,g,Hv
    OUTPUTS:
        f: objective function value
        g: gradient of objective function
        Hv: a Hessian-vector product function handle of any column vector v
    """
    
    def __init__(self, X, Y, w, HProp=None):
        self.X = X
        self.n, self.d = X.shape
        self.Y = Y
        self.HProp = HProp    
        self.C = int(len(w)/d)
        w = w.reshape(d*C,1) #[d*C x 1]
        W = w.reshape(C,d).T #[d x C]
        XW = torch.mm(X,W) #[n x C]
        large_vals = torch.max(XW,axis = 1)[0] #[n,1 ]
        large_vals = torch.clamp(large_vals, min=0) #M(x), [n, 1]
        #XW - M(x)/<Xi,Wc> - M(x), [n x C]
        XW_trick = XW - large_vals.repeat(C, 1).T
        #sum over b to calc alphax, [n x total_C]
        XW_1_trick = torch.cat((-large_vals.reshape(-1,1), XW_trick), 1)
        #alphax, [n, ]
        sum_exp_trick = torch.sum(torch.exp(XW_1_trick), axis = 1)
        log_sum_exp_trick = large_vals + torch.log(sum_exp_trick)  #[n, 1]
        self.fx = torch.sum(log_sum_exp_trick) - torch.sum(
                torch.sum(XW*self.Y,axis=1))
        inv_sum_exp = 1./sum_exp_trick
        inv_sum_exp = inv_sum_exp.repeat(C, 1).T
        self.S = inv_sum_exp*torch.exp(XW_trick) #h(x,w), [n x C] 
        if self.HProp == None:
            self.subX = self.X
            self.subn = self.n
            self.subS = self.S
        else:
            self.subn = np.int(np.floor(self.n*self.HProp))
            idx_H = np.random.choice(self.n, self.subn, replace = False)
            self.subX = self.X[idx_H,:]
            self.subS = self.S[idx_H,:]
        
    def f(self):
        return self.fx/self.n

    def g(self):        
        gx = torch.mm(self.X.T, self.S-self.Y)/self.n #[d x C]
        return gx.T.flatten()
        
    def Hv(self, v):
        v = v.reshape(len(v),1)
        V = v.reshape(self.C, self.d).T #[d x C]
        A = torch.mm(self.subX,V) #[n x C]
        AS = torch.sum(A*self.subS, axis=1)
        rep = AS.repeat(C, 1).T #A.dot(B)*e*e.T
        XVd1W = A*self.subS - self.subS*rep #[n x C]
        Hv = torch.mm(self.subX.T, XVd1W)/self.subn #[d x C]
        Hv = Hv.T.flatten() #[d*C, ] #[d*C, ]
        return Hv

#@profile
def main():        
    torch.manual_seed(0)
    import torch.utils.data as data
    import torchvision.datasets as datasets
    from  torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from time import time
    total_C = 10
    data_dir = '../Data'
    train_Set = datasets.CIFAR10(data_dir, train=True,
                                transform=transforms.ToTensor(), 
                                download=True)  
    (n, r, c, rgb) = train_Set.data.shape
    d = rgb*r*c
    X = train_Set.data.reshape(n, d)
    X = X/255
    total_C = 10
    X = torch.DoubleTensor(X)
    Y_index = train_Set.targets  
    Y_index = torch.DoubleTensor(Y_index)
#    
#    train_Set = datasets.MNIST(data_dir, train=True,
#                                transform=transforms.ToTensor())
#    (n, d, d) = train_Set.data.shape
#    d = d**2
#    X = train_Set.data.reshape(n, d)
#    X = X/255
#    Y_index = train_Set.targets
    I = torch.eye(total_C, total_C - 1)
    Y = I[np.array(Y_index), :]
#    l = d*(total_C - 1)
#    lamda = 0
    w = torch.randn(d*(total_C-1), dtype=torch.float64)
    # reg = None
#    reg = lambda x: regConvex(x, lamda)
    HProp = 1
    tm = time()
    
    fun = lambda x, arg=None: softmax(X,Y,x, HProp, arg=arg)
    derivativetest(fun,w)    
    
#    fun = lambda x: softmax_class(X,Y,x, HProp)
#    derivativetest_class(fun,w)    
    print('time', time() - tm)

if __name__ == '__main__':
    main()