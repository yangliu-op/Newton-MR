import torch

def cappedCR(A, b, maxiter, tol, epsilon=0):
    x = torch.zeros_like(b)
    r = b.detach()
    d = r
    rel_res = torch.norm(r)/torch.norm(b)   
    Ab = tAx(A, b, epsilon)
    # rtol = epsilon*tol
    T = 0
    Ar = Ab
    Ad = Ar
    Ax = x.clone().detach()
    d_type = 'SOL'
    b_norm = torch.norm(b)
    r_norm = b_norm
    
    while T < maxiter:
        T += 1
        rAr = torch.dot(r, Ar)
#        print('rAr',rAr/torch.norm(r)**2)
#        print('rAr',rAr, r_norm**2, rAr/r_norm**2, epsilon)
        if rAr/r_norm**2 < epsilon:
            d_type = 'NC'
            return r, rAr, T, d_type
        alpha = rAr/torch.norm(Ad)**2
        x = x + alpha*d
#        print(x.T.dot(Ax(A, -b)))
        r_new = r - alpha*Ad
        r_norm = torch.norm(r_new)
#        if torch.norm(r_new) - epsilon*tol*torch.norm(x) < 0:
        if torch.dot(x, r_new) - epsilon*tol*torch.norm(x)**2 < 0:
#        if torch.dot(x, r_new) < 0:
#            print('x.T r, Ture', rel_res, T)
            return x, rel_res, T, d_type
#        if torch.norm(r_new)- epsilon*tol*torch.norm(x) < 0:
#            print('torch.norm(r), Ture', rel_res, T)
#            return x, rel_res, T, d_type
#        print('norm', torch.norm(r_new), torch.norm(x), torch.norm(r_new)*torch.norm(x))
        Ar_new = tAx(A, r_new, epsilon)
        rAr_new = torch.dot(r_new, Ar_new)
        beta = rAr_new/rAr
        r = r_new # update variables
        # r_new = 0
#        print('123',beta*d.T.dot(tAx(A, r, epsilon)), d.T.dot(r))
        Ar = Ar_new
        # Ar_new = 0
        rel_res = torch.norm(r)/torch.norm(b)
#        d_old = d
        d = r + beta*d
#        Ad_old = Ad
        Ad = Ar + beta*Ad
        dAd = torch.dot(d, Ad)
#        print('dAd',dAd/torch.norm(d)**2)
        if dAd/torch.norm(d)**2 < epsilon:
            d_type = 'NC'
#            print('dAd')
            return d, dAd, T, d_type
        Ax = Ax + alpha*Ad
        xAx = torch.dot(x, Ax)
        if xAx/torch.norm(x)**2 < epsilon:
            d_type = 'NC'
#            print('xAx')
            return x, xAx, T, d_type
#    print('x.T r, Ture', rel_res, T)
    return x, rel_res, T, d_type

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
    # print(zeta, nu)
    # tau_r = np.sqrt(tau_r)
    
    while nu >= tau_a + tau_r * norm_g and T < maxiter:
        T += 1
        # print(delta/pi, zeta/rho)
        if delta <= epsilon*pi or zeta <= epsilon*rho:
            if T == 1:
#                print('-g')
                return b, rel_res, T, 'NC'
            else:
#                print(delta/pi, zeta/nu)
                return s, rel_res, T, 'Sol'
        alpha = zeta/torch.norm(q)**2
        s = s + alpha*p
        # print(T, torch.norm(s))
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
        # print(rel_res)
    return s, rel_res, T, 'Sol'

def CR(H, g, tau_a, tau_r, maxiter):
    T = 0
    s = torch.zeros_like(g)
    r = -g
    u = Ax(H, r)
    zeta = torch.dot(r, u)
    p = r
    q = u
    rho = torch.dot(r, r)
    nu = torch.sqrt(rho)
    norm_g = nu
    
    while nu >= tau_a + tau_r * norm_g and T < maxiter:
        T += 1
        alpha = zeta/torch.norm(q)**2
        s = s + alpha*p
        r = r - alpha*q
        rho = rho - alpha*zeta #rho = torch.norm(r)^2
        nu = torch.sqrt(rho) #nu = torch.norm(r)
        u = Ax(H, r)
        zeta_new = torch.dot(r, u) #zeta = r'Hr
        beta = zeta_new/zeta
        zeta = zeta_new
        p = r + beta * p
        q = u + beta*q #q = Hp
        # print(torch.norm(g+Ax(H,s))/torch.norm(g))
    return s, T    
    
def myCR(A, b, tol, maxiter, x0=None):
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0
    r = b - Ax(A, x)
    T = 0
    rel_res = torch.norm(r)/torch.norm(b)        
    p = r.clone()
    Ar = Ax(A, r)
    Ap = Ar
    # Vn = Ap/torch.norm(Ap)
    # X = x
#    xTr = 0
    
    while T < maxiter and rel_res > tol:
        T += 1
        rAr = torch.dot(r, Ar)
        pAAp = torch.dot(Ap, Ap)
        alpha = rAr/pAAp
#        print(xTr + alpha*p.T.dot(b) - alpha**2 * p.T.dot(Ap))
#        print(xTr + alpha*p.T.dot(r - Ax(A,x)) - alpha**2 * p.T.dot(Ap))
#        print('pb', p.T.dot(r - Ax(A,x)) - alpha * p.T.dot(Ap))
        x = x + alpha*p
        # print('in Span(Ap)', torch.norm(torch.mv(Vn, torch.mv(Vn.T, x)) - x), 
        #     torch.norm(torch.mv(Vn, torch.mv(Vn.T, Ax(A, x)-b))))
        # print('CR', T, x[:,0])
#        print('CR', alpha, torch.norm(p), torch.norm(x))
        r_new = r - alpha*Ap
#        oldpTp =pTr
#        xTr = r_new.T.dot(x)
#        print('pTr', xTr>0, xTr, rel_res)
#        print('pTg', -b.T.dot(x)<0, -b.T.dot(x)) 
#        print(torch.dot(x, r_new), torch.dot(x, r_new) - 1E-3*torch.dot(x, x), torch.norm(r_new))
#        print('P', torch.norm(x), torch.norm(b)/5)
        Ar_new = Ax(A,r_new)
        rAr_new = torch.dot(r_new, Ar_new)
        beta = rAr_new/rAr
        r = r_new # update variables
        Ar = Ar_new
        rel_res = torch.norm(r)/torch.norm(b)
        p = r + beta*p
        Apl = Ap
        Ap = Ar + beta*Ap
        # print('Ortho&&&&&&&&&&',torch.dot(Ap, Apl))
        # X = torch.cat((X, x.reshape(-1,1)), 1)
        # Vn = torch.cat((Vn, Ap.reshape(-1,1)/torch.norm(Ap)), 1)
#        print('normr',torch.norm(r))
    return x, rel_res, T
    # return x, rel_res, T, Vn, X

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

def main():
    from scipy.linalg import inv, svd, eig, norm
    from myMRII import myMINRES, myMRII
    from MinresQLP import MinresQLP
    from myCG import myCG
# =============================================================================
    from time import time
    # torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from myMINRES import myMINRES
    # device = torch.device("cpu")
    n = 50
    A = torch.randn(n, n, device=device, dtype=torch.float64)
    A = torch.mm(A.T, A)
    b = torch.randn(n, device=device, dtype=torch.float64)
    t1 = time()
    print(cappedCR(A + A.T, b, 100, 1E-2))
    print(' ')
    print(modifiedCR(A + A.T, b, 100, 0, 1E-2))
    print(' ')
    print(myMINRES(A + A.T, b, 1E-2, 100))
    print(' ')
    print(CR(A + A.T, -b, 0, 1E-2, 100))
    print(' ')
    print(myCR(A + A.T, b, 1E-2, 100))
    print(' ')
    print(time() - t1)
# =============================================================================
    # A = np.diag([2, -5, 3])
    # d = 3
    # n = 3
    # b = np.ones((d,1))
# =============================================================================
#     A = np.diag([100,2,111,4,0.001])
#    d = 8
#    n = 6
#    B = np.diag([1E2,1E3,1,-1E-2,-1E5,0,0,0])
##    B = np.diag([1E2,1E3,1,1E-2,1E5, 0, 0, 0])
##    B = np.diag([1E2,-2E3,0])
#    C = np.random.randn(d,d)
#    Uf, S, V = svd(C + C.T)
#    A = Uf.dot(B.dot(V))
#    U = Uf[:,:-3]
#    b = np.random.randn(d,1)
#    b = U.dot(U.T.dot(b))
# =============================================================================
# #    print(U.shape)
# #    U = Uf[:,:-1]
# #    print(svd(A + A.T)[1])
#     rtol = 1E-9
# #    eig = np.linspace(-4,10,d)
# #    A = np.diag(eig)
# #    b = np.zeros((d,1))
# #    b = np.ones((d,1))
# #    b = bf
# #    t0 = time.time()
#     x = MinresQLP(A, b, rtol, n)[0]
#     x0 = myMRII(A, b, rtol, n, negCurve=False, reorth=True)[0]
# #    x1 = myMINRES(A, b, rtol, n, negCurve=False)[0]
#     x1f = myMINRES(A, b, rtol, n, negCurve=True, reorth=True)
#     x1 = x1f[0]
#     V1 = x1f[-2]
#     x2f = myCR(A, b, rtol, n)
#     x2 = x2f[0]
#     V2 = x2f[-2]
#     print(' ')
# #    x3 = myCG(A, b, rtol, n)[0]
#     print(x[:,0])
#     print(x0[:,0])
#     print(x1[:,0])
#     print(x2[:,0])
#     print(' ')
# #    print(x3[:,0])
#     print(' ')
#     r = b - Ax(A, x)
#     r0 = b - Ax(A, x0)
#     r1 = b - Ax(A, x1)
#     r2 = b - Ax(A, x2)
# #    r3 = b - Ax(A, x3)
# #    print(torch.norm(r), torch.norm(r0), torch.norm(r1), torch.norm(r2), torch.norm(r3))
# #    print(test[0], inv(A).T.dot(b))
# =============================================================================
#     x, rel_res, T = test
#     print('myCR', 'x=', x, 'rel_res=', rel_res,'T=', T)
#     tau_r = 1E-2
# #    print('t=', t1 - t0)
#     p, rel_res, T, d_type = cappedCR(A, -b, d, tau_r)
# #    print(x)
#     print('cappedCR', 'p=', p, 'T=', T,d_type)
# #    x, T = CR(A, -b, 0,tau_r, d)
# ##    print(x)
# #    print('CR', 'x=', x, 'T=', T)
# #    t = 0.1
# #    tau_r = min(1, torch.norm(b)**t)
#     p, relres, T, dtype = modifiedCR(A, -b, 0,tau_r,0, d)
#     print(p.T.dot(Ax(A, p)))
# #    print(x)
#     print('modifiedCR', 'x=', x, 'T=', T)
# =============================================================================
    
if __name__ == '__main__':
    main()
