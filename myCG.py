import torch
    
    
def myCG(A, b, tol, maxiter):
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
        T += 1
        Ap = Ax(A, p)
        pAp = torch.dot(p, Ap)
        if pAp < 0:
            print('pAp =', pAp)
            raise ValueError('pAp < 0 in myCG')
        alpha = delta/pAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = torch.norm(r)/torch.norm(b)            
        if rel_res_best > rel_res:
            rel_res_best = rel_res
        prev_delta = delta
        delta = torch.dot(r, r)
        p = r + (delta/prev_delta)*p
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
    dNorm = 'Equal'
    T = 0
    # mp0 = 0
    flag = -2
    while T <= maxIter:
        Ad = Ax(A, d)
        T = T + 1
        dAd = torch.dot(d, Ad)
        if dAd <= 0:
            dType = 'NC'
            print('NC')
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
            flag = 0
            # print(mp)
            return x, rel_res, T, mp, dType, dNorm, flag
        rl = r
        r = rl + alpha*Ad
        norm_r2l = norm_r2
        norm_r = r.norm()
        norm_r2 = norm_r**2
        rel_res = norm_r2/norm_b
        # if rel_res > 1:
        #     print('rel_res', rel_res) # What the fk
        # if rel_res <= rtol:
        if norm_r <= rtol*norm_z:
            dNorm = 'Less'
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
            # print('NC')
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
        norm_r = r.norm()
        norm_r2 = norm_r**2
        rel_res = torch.sqrt(norm_r2/(norm_b**2))
        # if rel_res > 1:
        #     print('rel_res', rel_res) # What the fk
        if norm_r <= rtol*min(norm_b, epsilon*norm_z)/2:
            mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
            return z, rel_res, T, mp, dType
        beta = norm_r2/norm_r2l
        dl = d
        d = -r + beta*dl
        norm_d = torch.norm(d)
    mp = torch.dot(z, -b) + torch.dot(z, Ax(A, z))/2
    return z, rel_res, T, mp, dType

def CappedCG(H, b, zeta, epsilon, maxiter, rec, M=0):
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
        # print('b')
        return d, dType, j, ptHp, 1, rec
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
        if rec:
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
        if norm_Hp> M*norm_p:
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
        if norm_Hr> M*norm_r:
            M = norm_Hr/norm_r         
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        ytHy = torch.dot(y, tHy)
        ptHp = torch.dot(p, tHp)
        if ytHy < epsilon*norm_y**2:
            d = y
            dType = 'NC'
            # print('y')
            return d, dType, j, ytHy, relres, rec
        elif norm_r < tzeta*norm_g:
            # print('relres', relres)
            d = y
            return d, dType, j, 0, relres, rec
        elif ptHp < epsilon*norm_p**2:
            d = p
            dType = 'NC'
            # print('p')
            return d, dType, j, ptHp, relres, rec
        elif norm_r > torch.sqrt(T*tau**j)*norm_g:
            if rec: # regenerate the current iterates and record tHy matrix 
                # print('what')
                alpha_new = rr/ptHp
                y_new = y + alpha_new*p            
                tHy_new = tHy + alpha_new*tHp
                for i in range(j):
                    dy = y_new - Y[:, i]
                    dtHy = tHy_new - tHY[:, i]
                    dytHdy = torch.dot(dy, dtHy)
                    if dytHdy < epsilon*torch.norm(dy)**2:
                        d = dy
                        dType = 'NC'
                        print('dy')
                        return d, dType, j, dytHdy, relres, rec
            else:
                rec = True
                return d, dType, j, dytHdy, relres, rec
    print('Maximum iteration exceeded!')
    return y, dType, j, ytHy, relres, rec

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
    T = 4*kappa**4/(1 - torch.sqrt(tau))**2
    return kappa, tzeta, tau, T    
