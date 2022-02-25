import numpy as np
# import numpy.random as rand
from derivativetest import derivativetest
from logistic import logistic
# from regularizer import regConvex, regNonconvex
# from scipy.sparse import spdiags, csr_matrix
import torch
        
#global mydtype
#mydtype = torch.float64

def least_square(X, y, w, HProp=1, arg=None, reg=None, act='logistic'):
    """
    Least square problem sum(phi(Xw) - y)^2, where phi is logistic function.
    INPUT:
        X: data matrix
        y: lable vector
        w: variables
        HProp: subsampled(perturbed) Hessian proportion
        arg: output control
        reg: regularizer control
        act: activation function
    OUTPUT:
        f, gradient, Hessian-vector product/Gauss_Newton_matrix-vector product
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(w)
        
    n, d = X.shape
    # X = csr_matrix(X)    
    if act == 'logistic':
        fx, grad, Hess = logistic(X, w)
        
    #output control with least computation
    f = torch.sum((fx-y)**2)/n + reg_f
    if arg == 'f':        
        return f    
    
    g = 2*torch.mv(X.T, grad*(fx-y))/n + reg_g
        
    if arg == 'g':        
        return g
        
    if arg == 'fg':        
        return f, g
    
    if arg == None:
        if HProp == 1:
            #W is NxN diagonal matrix of weights with ith element=s2
            Hess = 2*(grad**2 + Hess*(fx-y))/n
            # W = spdiags(Hess.T[0], 0, n, n)
            # XW = X.T.dot(W) torch.mm()
            Hv = lambda v: hessvec(X, Hess, v) + reg_Hv(v)
            return f, g, Hv
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)       
#            Hess = 2*(grad**2 + Hess*(fx-y))/n
            # W = spdiags(Hess.T[0], 0, n, n)
            # XW = X.T.dot(W)
#                fullHv = lambda v: hessvec(XW, X, v) + reg_Hv(v)
            Hess = 2*(grad_H**2 + Hess_H*(fx_H-y[idx_H,:]))/n
            # W = spdiags(Hess.T[0], 0, n_H, n_H)
            # XW = X[idx_H,:].T.dot(W)
            Hv = lambda v: hessvec(X[idx_H,:], Hess, v) + reg_Hv(v)
            return f, g, Hv
    
    if arg == 'gn': #hv product        
        if HProp == 1:
            Hess_gn = 2*grad**2/n
            # W = spdiags(Hess_gn.T[0], 0, n, n)
            # XW = X.T.dot(W)
            Hv = lambda v: hessvec(X, Hess_gn, v) + reg_Hv(v)
            return f, g, Hv    
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)          
            Hess_gn = 2*grad_H**2/n_H
            # W = spdiags(Hess_gn.T[0], 0, n_H, n_H)
            # XW = X[idx_H,:].T.dot(W)
            Hv = lambda v: hessvec(X[idx_H,:], Hess_gn, v) + reg_Hv(v)
            return f, g, Hv
    
def hessvec(X, Hess, v):
    Xv = torch.mv(X, v)
    Hv = torch.mv(X.T, Hess*Xv)
    return Hv

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
    X = X/255
    X = X.double()
    Y_index = train_Set.targets
    Y = (Y_index>5)*torch.tensor(1).double()
    
#    print(Y)
    lamda = 0
    w = torch.randn(d, dtype=torch.float64)
    reg = None
#    reg = lambda x: regConvex(x, lamda)
    # reg = lambda x: regNonconvex(x, lamda)
    fun1 = lambda x, arg=None: least_square(X,Y,x,act='logistic', HProp=1, arg=arg,reg = reg)
    derivativetest(fun1,w)    
#    
if __name__ == '__main__':
    main()