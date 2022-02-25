import torch

def logistic(X, w, arg = None):
    """
    Logistic function
    INPUT:
        X: Matrix
        w: variable vector
        arg: output control
    OUTPUT:
        f, gradient, Hessian
    """
    #s = 1/(1+ np.exp(-t)) =  e^t/(1+e^t) = e^(t-M)/a
    #a = e^(-M)+e^(t-M)
    n, d = X.shape
    # w = w.reshape(d,1) 
    t = torch.mv(X, w)
    M = torch.max(torch.zeros_like(t),t)
    # power = torch.cat((-M, t-M), 0)
    # a = torch.sum(torch.exp(power), axis = 0)
    a = torch.exp(-M) + torch.exp(t-M)
    s = torch.exp(t-M)/a
    
    if arg == 'fx':
        return s
    
    if arg == 'grad':
        g = s*(1-s)
        return g
    
    if arg == 'Hess':
        H = s*(1-s)*(1-2*s)
        return H
    
    if arg == None:
        g = s*(1-s)
        H = s*(1-s)*(1-2*s)
        return s, g, H


def logit(t, arg = None):
    """
    Logistic function with out linear predictor
    """
    # d = len(w)
    # w = w.reshape(d,1) 
    # t = w
    M = torch.max(torch.zeros_like(t),t)
    # power = torch.cat((-M, t-M), 1)
    # a = torch.sum(torch.exp(power), axis = 1)
    a = torch.exp(-M) + torch.exp(t-M)
    s = torch.exp(t-M)/a
    print(s, s[0])
    # s = s[0]
    
    if arg == 'fx':
        return s
    
    if arg == 'grad':
        g = s*(1-s)
        return g
    
    if arg == 'Hess':
        H = s*(1-s)*(1-2*s)
        return H
    
    if arg == None:
        g = s*(1-s)
        H = s*(1-s)*(1-2*s)
        return s, g, H

def logit_rho(w, rho, arg = None):
    """
    Tool function for GMM model, s.t., 
    logit(t, rho) = f1/(rho*f1+(1-rho)*f2), 
    logit(-t, 1-rho) = f2/(rho*f1+(1-rho)*f2)
    """
    if abs(rho - 1) < 1e-15:
        return torch.ones_like(w)
    # if abs(rho) < 1e-15:
    #     return np.ones((len(w),1))
    d = len(w)
    w = w.reshape(d,1) 
    t = w
    M = torch.maximum(0,t)
    power = torch.cat((-M, t-M), 1)
    # M = np.maximum(0,t)
    # power = np.append(-M, t-M, axis = 1)
    ep = torch.exp(power)
    # print(ep[1,:])
    ep[ep==1] = ep[ep==1]*rho/(1-rho) # f1 * rho, f2 * (1 - rho)
    a = torch.sum(torch.exp(power), axis = 1) + 1e-50
    # a = np.sum(ep, axis = 1).reshape(d, 1)
    s = torch.exp(t-M)/a/(1-rho) # check
    return s
        # print(' ')
    # ep2 = ep
    # if rho == 1:
    #     print('a')
    # ep[:,0] = ep[:,0]*rho/(1-rho)
    # a = np.sum(ep, axis = 1).reshape(d, 1)
    # s = np.exp(t-M)/a/(1-rho)
    # ep2[:,-1] = ep2[:,-1]*(1-rho)
    # ep2[:,0] = ep2[:,0]*rho
    # a2 = np.sum(ep2, axis = 1).reshape(d, 1)
    # s2 = np.exp(t-M)/a2
    # print('a', norm(s - s2))