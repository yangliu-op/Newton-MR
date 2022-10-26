import torch

# @profile
def modifiedCR(A, b, maxiter, epsilon, tau_r, tau_a=0):
    #output: solution/direction, iteration, d_type/rel_res
    T = 0
    s = torch.zeros_like(b)
    r = b # -gradient
    u = Ax(A, r)
    zeta = torch.dot(r, u)
    p = r
    q = u
    delta = zeta
    rho = torch.dot(r, r)
    mu = rho
    pi = rho
    nu = torch.sqrt(rho)
    norm_g = nu
    rel_res = 1
    
    while nu >= tau_a + tau_r * norm_g and T < maxiter:
        T += 1
        # print(delta/pi, zeta/rho)
        if delta <= epsilon*pi or zeta <= epsilon*rho:
            if T == 1:
#                print('-g')
                return b, rel_res, T, 'NC'
            else:
                return s, rel_res, T, 'Sol'
        alpha = zeta/torch.norm(q)**2
        s = s + alpha*p
        r = r - alpha*q
        # rho = rho - alpha*zeta 
        nu = r.norm()
        rho = nu**2
         # = torch.sqrt(rho) #nu = torch.norm(r)
        u = Ax(A, r)
        zeta_new = torch.dot(r, u) #zeta = r'Hr
        beta = zeta_new/zeta
        zeta = zeta_new
        p = r + beta * p
        # pi = rho + 2*beta*(mu - alpha*delta) + beta**2*pi
        pi = p.norm()**2
        # mu = rho + beta*(mu - alpha*delta) #mu = p'r
        # pi = 2*mu - rho + beta**2*pi #pi = torch.norm(p)**2
        # pi = p.norm()
        q = u + beta*q #q = Hp
        delta = zeta + beta**2*delta #delta = p'Hp
        rel_res = nu/norm_g
    return s, rel_res, T, 'Sol'


def Ax(A, v):
    if callable(A):
        Ax = A(v)
    else:
        Ax =torch.mv(A, v)
    return Ax

def tAx(A, v, epsilon):
    if callable(A):
        Ax = A(v) + 2*epsilon*v
    else:
        Ax = torch.mv(A, v) + 2*epsilon*v
    return Ax
