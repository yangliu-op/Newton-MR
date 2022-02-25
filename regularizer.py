import torch

def regConvex(x, lamda, arg=None):
    """
    Returns f, g, Hv of lamda*||x||^2
    """
    f = 0.5*lamda*torch.dot(x, x)   
    if arg == 'f':
        return f.detach()
    
    g = lamda*x    
    if arg == 'g':
        return g.detach()
    if arg == 'fg':
        return f.detach(), g.detach()
    
    Hv = lambda v: (lamda*v).detach()
    
    if arg is None:
        return f.detach(), g.detach(), Hv    
#def regL1(x, mu):
#    f = mu*x.norm(p=1)
#    return f.detach()
#
#def prox(x, mu, Lambda=None):
#    shrinkage = abs(x) - mu/Lambda
#    D_ind = (shrinkage <= 0)
#    prox = torch.sign(x)*max(shrinkage, 0)
#    return prox.detach(), D_ind
#    
#def invProx(x, x_minus_grad, mu_Lambda):
#    '''
#    Input: x
#    xg: x_minus_grad
#    mu_Lambda: mu/lambda
#    '''
#    z = x
#    z_ind = (abs(x)>0)
#    z[z_ind] = torch.sign(z[z_ind])*(abs(z[z_ind]) + mu_Lambda)
#    z[~z_ind] = max(-mu_Lambda, min(mu_Lambda, x_minus_grad))
#    return z.detach()
    
def regNonconvex(x, lamda, a=1, arg=None):
    """
    returns  f, g, Hv of lamda*a*x^2/(1+a*x^2)
    """
    ax2 = a*x**2
    f = lamda*sum(ax2/(1+ax2))  
    if arg == 'f':
        return f.detach()
    
    g = lamda*2*a*x/(1+ax2)**2    
    if arg == 'g':
        return g.detach()
    if arg == 'fg':
        return f.detach(), g.detach()  
    
    Hv = lambda v: (lamda*2*a*(1-3*ax2)/(1+ax2)**3*v).detach()
    
    if arg is None:
        return f.detach(), g.detach(), Hv   


class L1(object):
    def __init__(self, mu=1, lamda=1):
        # Note that Lambda = lambda * eye
        self.mu = mu
        self.lamda = lamda
        self.mu_lamda = self.mu/self.lamda

    def f(self, x):
        fx = self.mu*x.norm(p=1)
        return fx.detach()
    
    def prox(self, z):
        #shrinkage = abs(z) - self.mu_lamda
        prox = torch.sign(z)*torch.clamp(abs(z) - self.mu_lamda, min=0)
        return prox.detach()
    
    def D(self, z):
        D_ind = (abs(z) - self.mu_lamda > 0)
        return D_ind
        
    def invprox(self, x, grad_lamda):
        '''
        Input: x
        xg: x_minus_grad
        mu_Lambda: mu/lambda
        '''
        z = torch.clone(x)
        x_minus_grad = x - grad_lamda # x - g/lamda
        z_ind = (abs(x)>0)
        z[z_ind] = torch.sign(z[z_ind])*(abs(z[z_ind]) + self.mu_lamda)
        z[~z_ind] = torch.clamp(x_minus_grad[~z_ind], 
         min=-self.mu_lamda, max=self.mu_lamda)
        return z.detach()
    

class groupSparsity(object):
    def __init__(self, mu=1, lamda=1, group=[]):
        # Note that Lambda = lambda * eye Therefore D is symmetric
        self.mu = mu
        self.lamda = lamda
        self.group = group

    def f(self, x):
        fx = 0
        for i in range(len(self.group)-1):
            xi = x[self.group[i]:self.group[i+1]]
            fx = fx + self.mu[i]*xi.norm()
        return fx.detach()
    
    def prox(self, z):
        x = torch.clone(z)
        for i in range(len(self.group)-1):   
            zi = z[self.group[i]:self.group[i+1]]
            zi_norm = zi.norm()
            shrinkage = zi_norm - self.mu[i]/self.lamda[i]
#            print(zi_norm, shrinkage)
            x[self.group[i]:self.group[i+1]] = torch.clamp(
                    shrinkage, min=0)*zi/zi_norm
        return x.detach()
    
    def LDv(self, z, v):
        x = torch.clone(v)
        for i in range(len(self.group)-1):   
            zi = z[self.group[i]:self.group[i+1]]
            vi = v[self.group[i]:self.group[i+1]]/self.lamda[i]
            zi_norm = zi.norm()
            shrinkage = zi_norm - self.mu[i]/self.lamda[i]
            if shrinkage > 0:
                x[self.group[i]:self.group[i+1]] = self.mu[i]*self.lamda[
                        i]/zi_norm*(vi - torch.dot(zi, vi)/zi_norm**2*zi)
        return x.detach()
    
    def Dv(self, z, v):
        x = torch.clone(v)
        for i in range(len(self.group)-1):   
            zi = z[self.group[i]:self.group[i+1]]
            vi = v[self.group[i]:self.group[i+1]]
            zi_norm = zi.norm()
            shrinkage = zi_norm - self.mu[i]/self.lamda[i]
            if shrinkage > 0:
                x[self.group[i]:self.group[i+1]] = self.mu[i]/self.lamda[
                        i]/zi_norm*(vi - torch.dot(zi, vi)/zi_norm**2*zi)
        return x.detach()
        
    def LD(self, z):
        x = lambda v: self.LDv(z, v)
        return x
        
    def D(self, z):
        x = lambda v: self.Dv(z, v)
        return x
    
    def L_(self, z):
        d = len(z)
        L = torch.eye(d, device = z.device, dtype=z.dtype)
        for i in range(len(self.group)-1):   
            L[self.group[i]:self.group[i+1], self.group[i]:self.group[
                    i+1]] = self.lamda[i] * L[self.group[i]:self.group[
                            i+1], self.group[i]:self.group[i+1]]
        return L
        
    
    def D_(self, z):
        d = len(z)
        D = torch.eye(d, device = z.device, dtype=z.dtype)
        for i in range(len(self.group)-1):   
            zi = z[self.group[i]:self.group[i+1]]
            zi_norm = zi.norm()
            mu_lamda = self.mu[i]/self.lamda[i]
            shrinkage = zi_norm - self.mu[i]/self.lamda[i]
            if shrinkage > 0:
                D[self.group[i]:self.group[i+1], self.group[i]:self.group[
                        i+1]] = mu_lamda/zi_norm*(D[self.group[i]:self.group[
                                i+1], self.group[i]:self.group[
                                        i+1]] - torch.ger(zi, zi)/zi_norm**2)
        return D
    
    def LD_(self, z):
        d = len(z)
        D = torch.eye(d, device = z.device, dtype=z.dtype)
        for i in range(len(self.group)-1):   
            Di = self.lamda[i]*torch.eye(self.group[i+1]-self.group[
                    i], device = z.device, dtype=z.dtype)
            zi = z[self.group[i]:self.group[i+1]]
            zi_norm = zi.norm()
            mu_lamda = self.mu[i]/self.lamda[i]
            shrinkage = zi_norm - self.mu[i]/self.lamda[i]
            if shrinkage > 0:
                D[self.group[i]:self.group[i+1], self.group[i]:self.group[
                        i+1]] = mu_lamda/zi_norm*(
                        Di - self.lamda[i]*torch.ger(zi, zi)/zi_norm**2)
        return D
        
#    def invprox(self, x, grad_lamda):
#        '''
#        Input: x
#        xg: x_minus_grad
#        mu_Lambda: mu/lambda
#        '''
#        z = torch.clone(x)
#        x_minus_grad = x - grad_lamda # x - g/lamda
#        z_ind = (abs(x)>0)
#        z[z_ind] = torch.sign(z[z_ind])*(abs(z[z_ind]) + self.mu_lamda)
#        z[~z_ind] = torch.clamp(x_minus_grad[~z_ind], 
#         min=-self.mu_lamda, max=self.mu_lamda)
#        return z.detach()

def LFnzk(gk, x, z, lamda, group):
    Fn = torch.clone(z - x)
    for i in range(len(group)-1):   
        Fn[group[i]:group[i+1]] = Fn[group[i]:group[i+1]] + gk[
                group[i]:group[i+1]]/lamda[i]
    return Fn


def Fnzk(gk, x, z, lamda, group):
    Fn = torch.clone(gk)
    for i in range(len(group)-1):   
        Fn[group[i]:group[i+1]] = Fn[group[i]:group[i+1]] + lamda[i]*(
                z[group[i]:group[i+1]] - x[group[i]:group[i+1]])
    return Fn

def main():
#    torch.manual_seed(7)
    import copy
    n = 10
    d = 6
    A = torch.randn(d,d, dtype=torch.float64)
#    b = torch.ones(n, dtype=torch.float64)
    gk = torch.randn(d, dtype=torch.float64)
    # w = np.ones((d,1))
    z0 = torch.rand(d, dtype=torch.float64)
    z1 = torch.randn(d, dtype=torch.float64)*2
#    x = x0
#    mu = 0.01
#    lamda = 0.4
#    z = L1(mu, lamda).invprox(x0, gk/lamda)
#    x00 = L1(mu, lamda).prox(z)[0]
#    print((x00 - x0).norm())
#    print(x00-x0)
    
    group = 3
    mu=torch.rand(group, dtype=torch.float64)*1E-1
    lamda=torch.rand(group, dtype=torch.float64)*1E2
    groups = torch.floor(torch.linspace(0, len(z0), len(mu)+1)).int()
    phi = groupSparsity(mu, lamda, groups)
    x0 = phi.prox(z0)
    x1 = phi.prox(z1)
    LD = phi.LD(z0)
    D = phi.D(z0)
    mLD = phi.LD_(z0)
    mD = phi.D_(z0)
    mL = phi.L_(z0)
#    x2 = torch.randn(d, dtype=torch.float64)
    Uf, s, V = torch.svd(mLD)
    ind = s > 1E-14
    U = Uf[:, ind]
    Up = Uf[:, ~ind]
    UU = U @ U.T
    print(U.shape)
    if Up.shape[1] != 0:
        UUp = Up @ Up.T
    xk = x1 - x0
    zk = z1 - z0
#    print(zk.norm(), LD(zk).norm())
    
    Ufs, ss, Vs = torch.svd(A + A.T)
    r = 5
    U1 = Ufs[:, :r]
    S = Ufs[:, :r] @ torch.diag(ss[:r]) @ Vs[:, :r].T
    pL = torch.inverse(mL)
#    B = U1 @ torch.diag(s1) @ V1.T
#    BD =  B@mD
#    M = BD + mL - mLD
#    U2 = Uf[:, :4]
#    UU2 = U2 @ U2.T
#    B = mL - mL @ torch.pinverse(mD)
#    M = B @ mD + mL - mLD
#    LM0 = pL @ M
    
    LM0 = mL @ (S + mD - torch.eye(d)) @ torch.pinverse(mD)
    U1, s1, V1 = torch.svd(LM0)
    ind1 = (s1 > 1E-10)
    is1 = 1/s1
    s1[~ind1] = 0
    is1[~ind1] = 0
    LM = U1 @ torch.diag(s1) @ V1.T
    pLM = V1 @ torch.diag(is1) @ U1.T
    
    
    DM0 = mD.T @ M
    DM1 = mLD @ LM
    U2, s2, V2 = torch.svd(DM0)
    ind2 = (s2 > 1E-15)
    is2 = 1/s2
    s2[~ind2] = 0
    is2[~ind2] = 0
    LM = U2 @ torch.diag(s2) @ V2.T
    pLM = V1 @ torch.diag(is2) @ U2.T
    
    
#    print(mD)
    s[s <= 1E-14] = 0
    s2 = torch.sqrt(s)
    sLD = Uf @ torch.diag(s2) @ V.T
    U1, s1, V1 = torch.svd(sLD)
    ind1 = (s1 > 1E-15)
    s1[~ind1] = 0
    LM = U1 @ torch.diag(s1) @ V1.T
    pLM = V1 @ torch.diag(torch.sqrt(s1)) @ U1.T
    
    
    U1, s1, V1 = torch.svd(LM0)
    ind1 = (s1 > 1E-15)
    s1[~ind1] = 0
    LM = U1 @ torch.diag(s1) @ V1.T
    pLM = V1 @ torch.diag(torch.sqrt(s1)) @ U1.T
    
    
    psLD = torch.pinverse(sLD)
    pDM = torch.pinverse(DM)
    pLM = torch.pinverse(LM)
#    print((DM - DM1).norm())
#    dd = torch.pinverse(psLD @ DM @ psLD) - sLD @ pDM @ sLD
    dd = torch.pinverse(sLD @ LM @ psLD) - sLD @ pLM @ psLD
#    dd = torch.pinverse(psLD @ DM) - pDM @ sLD
    dd2 = torch.pinverse(sLD @ LM) - pLM @ psLD
    print(dd.norm())
    print(dd2.norm())
#    Fn = Fnzk(gk, x0, z0, lamda, groups)
#    LFn = LFnzk(gk, x0, z0, lamda, groups)
#    print(Fn @ (x1 - x0), LFn @ (z1 - z0))
#    print(torch.dot(x2, LD(x2)))
#    print()
#    print(gk, torch.clamp(gk, max=0))
#    print(gk, torch.clamp(gk, max=1, min=-1))
if __name__ == '__main__':
    main()