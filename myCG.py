import torch
    
def myCG(A, b, tol, maxiter):
#def myCG(A, b, tol, maxiter, xx=None, L=None):
    """
    Conjugate gradient mathods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
    OUTPUT:
        x: best solution x
        rel_res: relative residual
        T: iterations
    """
    x = torch.zeros_like(b)
    r = b
    T = 0
    rel_res_best = 1
    rel_res = 1
        
    delta = torch.dot(r, r)
    p = r.clone()
    
    while T < maxiter and rel_res >= tol:
        # print(T)
        T += 1
        Ap = Ax(A, p)
        pAp = torch.dot(p, Ap)
#        if pAp < 0:
#            print('pAp =', pAp)
#            raise ValueError('pAp < 0 in myCG')
        alpha = delta/pAp
        xl = x
        x = x + alpha*p
#        xl1 = xl - ((A @ xl) - b)/L
#        xl2 = xl1 - ((A @ xl1) - b)/L
#        x1 = x - ((A @ x) - b)/L
#        print((x1 - xx).norm() - (xl2 - xx).norm())
        # print('xnorm', torch.norm(x), x[:,0])
        # print('in Span(p)', torch.norm(Vn.dot(Vn.T.dot(x)) - x))
        r = r - alpha*Ap
#        print('pTr', -b.T.dot(x)+0.5*torch.dot(x, Ax(A, x))<0, -b.T.dot(x)+x.T.dot(
#                Ax(A, x))<0, -b.T.dot(x)+torch.dot(x, Ax(A, x)))
#        print('pTg', -b.T.dot(x)<0, -b.T.dot(x))
#        print(' ')
        rel_res = torch.norm(r)/torch.norm(b)            
        if rel_res_best > rel_res:
            rel_res_best = rel_res
        prev_delta = delta
        delta = torch.dot(r, r)
        # p_old = p
        p = r + (delta/prev_delta)*p
    #     Vn = np.append(Vn, p/torch.norm(p), axis=1)
    #     X = np.append(X, x, axis=1)
    #     print('normp',torch.norm(p),torch.norm(r)**2/torch.norm(p))
    # return x, rel_res, T, Vn, X
    return x, rel_res, T
        
def SteihaugCG(A, b, rtol, maxIter, Delta=1):
    z = torch.zeros_like(b)
    g = -b
    r = g
    d = -r
    rel_res = 1
    norm_b = torch.norm(b)
    norm_z = 0
    norm_r2 = norm_b**2
    norm_d = norm_b
    dType = 'Sol'
    dNorm = 'Less'
    T = 0
    # mp0 = 0
    flag = -2
    while T <= maxIter:
        Ad = Ax(A, d)
        T = T + 1
        dAd = torch.dot(d, Ad)
        if dAd <= 0:
            dType = 'NC'
#            print('NC')
            flag = -1
            if T == 1:
                mp = - Delta*norm_b + dAd/norm_b**2/2
                return Delta*b/norm_b, rel_res, T, mp, dType, dNorm, flag
            else:
                zTd = torch.dot(z, d)
                diff = Delta**2 - norm_z**2
                tau = diff/(zTd + torch.sqrt(zTd**2 + norm_d**2*diff))
                x = z + tau*d
                mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
                dNorm = 'Equal'
                return x, rel_res, T, mp, dType, dNorm, flag
        alpha = norm_r2/dAd
        zl = z
        z = zl + alpha*d
        # mp0l = mp0
        # mp0 = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
        # print('mp', mp0)
        # if mp0 > mp0l:
        #     print('what')
        norm_zl = norm_z
        norm_z = torch.norm(z)
        if norm_z > Delta:
            zTd = torch.dot(zl, d)
            diff = Delta**2 - norm_zl**2
            tau = diff/(zTd + torch.sqrt(zTd**2 + norm_d**2*diff))
            x = zl + tau*d            
            mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
            dNorm = 'Equal'
            flag = 0
            # print(mp)
            return x, rel_res, T, mp, dType, dNorm, flag
        rl = r
        r = rl + alpha*Ad
        norm_r2l = norm_r2
        norm_r2 = torch.dot(r, r)
        rel_res = torch.sqrt(norm_r2/(norm_b**2))
        # if rel_res > 1:
        #     print('rel_res', rel_res) # What the fk
        if rel_res <= rtol:
            mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
            flag = 1
            return z, rel_res, T, mp, dType, dNorm, flag
        beta = norm_r2/norm_r2l
        dl = d
        d = -r + beta*dl
        norm_d = torch.norm(d)
    mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
    flag = 2
    return z, rel_res, T, mp, dType, dNorm, flag

def myCG_TR_Pert(A, b, rtol, maxIter, Delta=1, epsilon=1e-3):
    z =  torch.zeros_like(b)
    g = -b
    r = g
    d = -r
    rel_res = 1
    norm_b = torch.norm(b)
    norm_z = 0
    norm_r2 = norm_b**2
    norm_d = norm_b
    dType = 'Sol'
    # dNorm = 'Less'
    T = 0
    # mp0 = 0
    # flag = -2
    while T <= maxIter:
        tAd = Ax(A, d) + 2*epsilon*d
        T = T + 1
        dtAd = torch.dot(d, tAd)
        if dtAd <= epsilon*norm_d**2:
            dType = 'NC'
#            print('NC')
            # flag = -1
            if T == 1:
                mp = - Delta*norm_b + dtAd/norm_b**2/2
                return Delta*b/norm_b, rel_res, T, mp, dType
            else:
                zTd = torch.dot(z, d)
                diff = Delta**2 - norm_z**2
                tau = diff/(zTd + torch.sqrt(zTd**2 + norm_d**2*diff))
                x = z + tau*d
                mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
                # dNorm = 'Equal'
                return x, rel_res, T, mp, dType
        alpha = norm_r2/dtAd
        zl = z
        z = zl + alpha*d
        # mp0l = mp0
        # mp0 = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
        # print('mp', mp0)
        # if mp0 > mp0l:
        #     print('what')
        norm_zl = norm_z
        norm_z = torch.norm(z)
        if norm_z > Delta:
            zTd = torch.dot(zl, d)
            diff = Delta**2 - norm_zl**2
            tau = diff/(zTd + torch.sqrt(zTd**2 + norm_d**2*diff))
            x = zl + tau*d            
            mp = torch.dot(x, -b) + torch.dot(x, Ax(A, x))/2
            return x, rel_res, T, mp, dType
        rl = r
        r = rl + alpha*tAd
        norm_r2l = norm_r2
        norm_r2 = torch.dot(r, r)
        rel_res = torch.sqrt(norm_r2/(norm_b**2))
        # if rel_res > 1:
        #     print('rel_res', rel_res) # What the fk
        if rel_res <= rtol*min(norm_b, epsilon*norm_z)/2:
            mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
            return z, rel_res, T, mp, dType
        beta = norm_r2/norm_r2l
        dl = d
        d = -r + beta*dl
        norm_d = torch.norm(d)
    mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
    return z, rel_res, T, mp, dType

def CappedCG(H, b, zeta, epsilon, maxiter, M=0):
    g = -b
    y =  torch.zeros_like(g)
#    print(dim)
    kappa, tzeta, tau, T = para(M, epsilon, zeta)
    tHy = y.clone()
    tHY = y.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    r = g
    p = -g
    tHp = Ax(H, p) + 2*epsilon*p
    j = 1
    ptHp = torch.dot(p, tHp)
    norm_g = torch.norm(g)
    norm_p = norm_g
    rr = torch.dot(r, r)
    dType = 'Sol'
    relres = 1
    if ptHp < epsilon*norm_p**2:
        d = p
        dType = 'NC'
        print('b')
        return d, dType, j, ptHp, 1
    norm_Hp = torch.norm(tHp - 2*epsilon*p)
    if norm_Hp > M*norm_p:
        M = norm_Hp/norm_p
        kappa, tzeta, tau, T = para(M, epsilon, zeta)
    while j < maxiter:
#        print(j, M, torch.norm(r)/norm_g, tzeta)
        alpha = rr/ptHp
        y = y + alpha*p
        Y = torch.cat((Y, y.reshape(-1, 1)), 1) #record y
        norm_y = torch.norm(y)
        tHy = tHy + alpha*tHp
        tHY = torch.cat((tHY, tHy.reshape(-1, 1)), 1) # record tHy
        norm_Hy = torch.norm(tHy - 2*epsilon*y)
        r = r + alpha*tHp
        rr_new = torch.dot(r, r) 
        beta = rr_new/rr
        rr = rr_new
        p = -r + beta*p #calculate Hr
        norm_p = torch.norm(p)        
        tHp_new = Ax(H, p) + 2*epsilon*p #the only Hessian-vector product
        j = j + 1
        tHr = beta*tHp - tHp_new #calculate Hr
        tHp = tHp_new
        norm_Hp = torch.norm(tHp - 2*epsilon*p)
        ptHp = torch.dot(p, tHp)  
        if  norm_Hp> M*norm_p:
            M = norm_Hp/norm_p
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if norm_Hy > M*norm_y:
            M = norm_Hy/norm_y
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        norm_r = torch.norm(r)
        relres = norm_r/norm_g
#        print(norm_r/norm_g, tzeta)
        norm_Hr = torch.norm(tHr - 2*epsilon*r)
#        print(norm_r, torch.norm(H(y) + g))
        if  norm_Hr> M*norm_r:
            M = norm_Hr/norm_r         
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if torch.dot(y, tHy) < epsilon*norm_y**2:
            d = y
            dType = 'NC'
            # print('y')
            return d, dType, j, torch.dot(y, tHy), relres
        elif norm_r < tzeta*norm_g:
            # print('relres', relres)
            d = y
            return d, dType, j, 0, relres
        elif torch.dot(p, tHp) < epsilon*norm_p**2:
            d = p
            dType = 'NC'
            # print('p')
            return d, dType, j, torch.dot(p, tHp), relres
        elif norm_r > torch.sqrt(T*tau**j)*norm_g:
            print('what')
            alpha_new = rr/ptHp
            y_new = y + alpha_new*p            
            tHy_new = tHy + alpha_new*tHp
            for i in range(j):
                dy = y_new - Y[:, i]
                dtHy = tHy_new - tHY[:, i]
                if torch.dot(dy, dtHy) < epsilon*torch.norm(dy)**2:
                    d = dy
                    dType = 'NC'
                    print('dy')
                    return d, dType, j, torch.dot(dy, dtHy), relres
    print('Maximum iteration exceeded!')
    return y, dType, j, 0, relres


def myCG_truncated(A, b, tol, maxiter, x0=None):
    x = torch.zeros_like(b)
    r = -b
    T = 0
    norm_b = torch.norm(b)        
    delta = torch.dot(r, r)
    d = -r
    epsilon = min(0.5, torch.sqrt(norm_b))*norm_b
    while T < maxiter and torch.norm(r) > epsilon:
        T += 1
        Ad = Ax(A, d)
        dAd = torch.dot(d, Ad)
        if dAd <= 0:
            if T == 0:
                return b, T
            else:
                return x, T
        alpha = delta/dAd
        x = x + alpha*d
        r = r + alpha*Ad
        prev_delta = delta
        delta = torch.dot(r, r)
        d = -r + (delta/prev_delta)*d
    return x, T

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
        Ax = torch.mv(A + 2*epsilon*torch.eye(len(v)),v)
    return Ax
    
def para(M, epsilon, zeta):
    # if torch.tensor(M):
    #     M = M.item()
    kappa = (M + 2*epsilon)/epsilon
    tzeta = zeta/3/kappa
    # print('kappa', kappa)
    sqk = torch.sqrt(torch.tensor(float(kappa)))
    tau = sqk/(sqk + 1)
    T = 4*kappa**4/(1 + torch.sqrt(tau))**2
    return kappa, tzeta, tau, T    

def main(): 
# =============================================================================
    from time import time
    # torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n = 500
    A = torch.randn(n, n, device=device, dtype=torch.float64)
    b = torch.randn(n, device=device, dtype=torch.float64)
#    t1 = time()
    B = A.T @ A
    LmaxB = max(torch.eig(B)[0][:,0])
    x = myCG(B, -b, 1E-9, n)
#    x = myCG(B, -b, 1E-9, n, xx=-torch.inverse(B) @ b, L=LmaxB)
    print(' ')
#    print(SteihaugCG(A + A.T, b, 1E-2, 100))
#    print(' ')
#    print(myCG_TR_Pert(A + A.T, -b, 1E-2, 1e-3, 100))
#    print(' ')
#    print(CappedCG(A + A.T, b, 1E-2, 1e-3, 100))
#    print(' ')
#    print(CappedCG_old(A + A.T, b, 1E-2, 1e-3, 100))
#    print(' ')
#    print(time() - t1)
#     import numpy as np
#     from MinresQLP import MinresQLP
#     B = np.diag([1,2,3,4,5])
#     epsilon = 1E-1
#     A = lambda v: B.dot(v)
#     tA = lambda v: B.dot(v) + 2*epsilon*v
#     b = np.ones((5,1))
# #    U, s, Vt = svd(A)
# #        U = U[:,:len(s)]
# #        print(b - U.dot(U.T.dot(b)))
# #    x1 = MinresQLP(A, b, 1E-4, 50, MR2=True)[0]
# #        x2 = MinresQLP(A, b, 1E-4, 50, MR2=True)[0]
# #        x2 = MinresQLP(A.dot(A), A.dot(b), 1E-40, 5)[0]
# #    x3 = minres(A.dot(A), A.dot(b), x0=np.zeros((5,1)), tol=1E-4, maxi
#     print('CG')
#     x1 = myCG(tA, b, tol=1E-6, maxiter=5)
#     print(' ', x1)
#     print('CappedCG')
# #    epsilon = 0
#     x2 = CappedCG(A, b, 1E-6, epsilon, maxiter=5)
#     print(' ', x2)
#     # x = cg(A,b,x0=b,tol=1e-12,maxiter=None,callback=cb)[0]
    
if __name__ == '__main__':
    main()