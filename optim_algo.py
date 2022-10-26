"""
Optimisation algorithms,

    INPUT:
        obj: function handle of objective function, gradient and 
            Hessian-vector product
        x0: starting point
        HProp: subsampled(perturbed) Hessian proportion
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        innerSolverTol: inexactness tolerance of Inner-solver MINRES-QLP
        lineSearchMaxItrs: maximum iteration of line search
        gradTol: stopping condition para s.t., torch.norm(gk) <= Tol
        beta: parameter of Armijo line-search
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, step-size, direction-type]

Termination condition: norm(gradient) < gradtol. 
Otherwise either reach maximum iterations or maximum oracle calls
""" 

import torch
from lbfgs import lbfgs
from myCG import CappedCG, SteihaugCG, myCG_TR_Pert
from time import time
from linesearch import (Ax, linesearchgrad, linesearchzoom, 
                        linesearchNC, linesearch_NewtonMR, 
                        linesearch_NewtonMR_NC, linesearch_NewtonCG_Pert_NC,
                        )
from myMINRES import myMINRES
from MinresQLP import MinresQLP
from myCR import modifiedCR
import copy
     
global num_every_print, is_zero, orcl_every_record
num_every_print = 1E0
orcl_every_record = 6E11
is_zero = 1E-18
        
#global mydtype
def myPrint(fk, gk_norm, orcl, iters, tmk, 
            alphak=0, iterLS=0, iterSolver=0, rel_res=0, dType=None, sparsity=None):
    """
    A print function for every iteration.
    """
    if iters%(num_every_print*10) == 0:
        prthead1 = '  iters  iterSolver   iterLS   Time      f         ||g||'
        prthead2 = '      alphak      Prop   Relres    Direction   Sparsity'        
        prt = prthead1 + prthead2
        print(prt)
    
    prt1 = '%8g %8g' % (iters, iterSolver)
    prt2 = '%8s %8.2f' % (iterLS, tmk)
    prt3 = ' %8.2e     %8.2e ' % (fk, gk_norm)
    prt4 = '   %8.2e %8g ' % (alphak, orcl)
    prt5 = '%8.2e %8s  ' % (rel_res, dType)
#    print(prt1, prt2, prt3, prt4, prt5)  
    if sparsity is not None:
        prt6 = ' %8.1f%%  ' % (sparsity*100)
        print(prt1, prt2, prt3, prt4, prt5, prt6) 
    else:
        prt5 = '%8.2e %8s  ' % (rel_res, dType)
        print(prt1, prt2, prt3, prt4, prt5)  
    
def termination(objVal, gradNorm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):  
    """
    termination condition
    """
    termination = False
    if gradNorm < gradTol or iters >= mainLoopMaxItrs or orcl >= funcEvalMax:
        termination = True
#    if (np.isnan(gradNorm) or np.isinf(gradNorm) or np.isnan(objVal) or np.isinf(objVal) or
#        gradNorm < gradTol or iters >= mainLoopMaxItrs or orcl >= funcEvalMax):
#        termination = True
        return termination
    
def recording(matrix, v1, v2, v3, v4, v5, NC=None):    
    """
    recording matrix with row [fx, gx, oracle-calls, time, step-size, direction-type]
    and iteration as columns
    """
    v = torch.tensor([v1, v2, v3, v4, 
                      v5], 
                     device = matrix.device, 
                     dtype=torch.float64)
    if NC is not None: 
        if NC == 'NC':
            v = torch.cat((v, torch.ones(1, device = matrix.device, 
            dtype=torch.float64)))
        else:
            v = torch.cat((v, torch.zeros(1,device = matrix.device, 
            dtype=torch.float64)))
    matrix = torch.cat((matrix, v.reshape(1,-1)), axis=0)
    return matrix


def orc_call(iterSolver, HProp, iterLS=None):
    if iterLS == None:
        iterLS = 0
    return 2 + 2*iterSolver*HProp + iterLS


def Newton_MR(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
                 innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
                 beta=0.25, epsilon=0, show=True, 
                 record_txt=None):
    iters = 0
    orcl = 0
    x = x0.clone() 
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()
    alphak = 1    
    alphal = 1
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    dType = 'None'
    
    # Initialize f0, g0, oracle_call0, time0, alpha0, NC
    record = torch.tensor([fk, gk_norm, 0, 0, 1, 0], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS, iterSolver, 
                    rel_res, dType)
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break

        t0 = time()
        p, rel_res, iterSolver, rk, dType =  myMINRES(Hk, -gk, innerSolverTol*epsilon, 
                                            innerSolverMaxItrs, reOrth=True)
        if torch.isnan(p).any():
            break
        if dType == 'Sol':
            x, alphak, iterLS = linesearch_NewtonMR(
                  obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, c1=beta, alpha0=1)
        else:
            p = rk
              # linesearch_NewtonMR_NC is two-direction linesearch
            x, alphak, iterLS = linesearch_NewtonMR_NC(
                  obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, c1=beta, alpha0=alphal
                  )
            alphal = alphak # set initial stepsize for the next iteration
        orcl += orc_call(iterSolver, HProp, iterLS)
            
        if alphak < is_zero:
            orcl = funcEvalMax
        else:
            fk, gk, Hk = obj(x)
            gk_norm = gk.norm()
            
        iters += 1  
        tmk += time()-t0

        record = recording(record, fk, gk_norm, orcl, tmk, alphak, dType)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('%s_%s' % ('Newton_MR', orcl_sh), record)
            orcl_sh = orcl_sh*10
    
    return x, record

def Naive_Newton_MR(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
                    innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
                    beta=1e-4, show=True, record_txt=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0) 
    x = copy.deepcopy(x0)
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()        
    alphak = 1
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, 1], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS, iterSolver, 
                    rel_res)
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    

        t0 = time()
        p, rel_res, iterSolver = MinresQLP(Hk, -gk, 
                                            rtol=innerSolverTol, 
                                            maxit=innerSolverMaxItrs)
        if torch.isnan(p).any():
            break
        
        x, alphak, iterLS = linesearchgrad(obj, torch.dot(gk, gk), Ax(Hk, gk), x, p, 
                                           lineSearchMaxItrs, 1, c1=beta)       
        
        orcl += orc_call(iterSolver, HProp, 2*iterLS)  
            
        iters += 1  
        fk, gk, Hk = obj(x)
        gk_norm = gk.norm()
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, alphak)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('Naive_Newton_MR_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record
        
       
def L_BFGS(obj, x0, mainLoopMaxItrs, funcEvalMax, lineSearchMaxItrs=50, 
           gradTol=1e-10, L=10, beta=1e-4, beta2=0.9, show=True,
           record_txt=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)
    fk, gk = obj(x, 'fg')
    gk_norm = gk.norm()    
    l= len(gk)   
    alphak = 1
    tmk = 0
    iterLS = 0
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, 1], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    
        
        t0 = time()  
        if iters == 0:
            p = -gk
            S = torch.empty(l, 0, device=gk.device, dtype=x0.dtype)
            Yy = torch.empty(l, 0, device=gk.device, dtype=x0.dtype)
        else:
            s = alphak_prev * p_prev
            y = gk - g_prev
            
            if S.shape[1] >= L:
                S = S[:,1:]
                Yy = Yy[:,1:]
            S = torch.cat((S, s.reshape(-1,1)), axis=1)
            Yy = torch.cat((Yy, y.reshape(-1,1)), axis=1)
            p = -lbfgs(gk, S, Yy)
#        
        #Strong wolfe's condition with zoom
        if torch.isnan(p).any():
            break
        x, alphak, iterLS, iterLS_orcl = linesearchzoom(
            obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, 
            c1=beta, c2=beta2, fe=funcEvalMax-orcl)
        
        g_prev = gk
        p_prev = p
        alphak_prev = alphak         

        
        fk, gk = obj(x, 'fg')
        gk_norm = gk.norm()
        iters += 1
        orcl += 2 + iterLS_orcl
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, alphak)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('L_BFGS_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record


def Nonlinear_CG(obj, x0, mainLoopMaxItrs, funcEvalMax, lineSearchMaxItrs=50, 
               gradTol=1e-10, beta=1e-4, beta2=0.1, show=True,
               record_txt=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)
    fk, gk = obj(x, 'fg')
    gk_norm = gk.norm()    
    alphak = 1
    tmk = 0
    iterLS = 0
    
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, 1], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    
        
        t0 = time()  
        if iters == 0:
            p = -gk
        else:
            # print(' ', iters)
            
            # # Fletcher-Reeves
            # beta = (gk @ gk)/(gk_prev @ gk_prev)
            
            # # Polak-Ribiere
            # beta = (gk @ (gk - gk_prev))/(gk_prev @ gk_prev)
            
            # # Polak-Ribiere +
            # beta = max((gk @ (gk - gk_prev))/(gk_prev @ gk_prev), 0)
            
            # # Hestenes-Stiefel
            # beta = -(gk @ (gk - gk_prev))/(p_prev @ gk - gk_prev)
            # # Dai-Yuan
            # beta = (gk @ gk)/(p_prev @ gk - gk_prev)
            
            # PR-FR
            betaFR = (gk @ gk)/(gk_prev @ gk_prev)
            betaPR = (gk @ (gk - gk_prev))/(gk_prev @ gk_prev)
            if betaPR < -betaFR:
                beta = -betaFR
            else:
                if abs(betaPR) <= betaFR:
                    beta = betaPR
                else:
                    if betaPR > betaFR:
                        beta = betaFR     
            
            p_prev = p
            p = -gk + beta*p_prev
            alphak = alphak*(gk_prev @ p_prev)/(gk @ p)
        
        #Strong wolfe's condition with zoom
        x, alphak, iterLS, iterLS_orcl = linesearchzoom(
            obj, fk, gk @ p, x, p, lineSearchMaxItrs, c1=beta, c2=beta2, alpha0=alphak)
        
        gk_prev = gk
        fk, gk = obj(x, 'fg')
        gk_norm = gk.norm()
        iters += 1
        orcl += 2 + iterLS_orcl
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, alphak)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('Nonlinear_CG_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record
        
# @profile
def Newton_CR(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, 
                 innerSolverTol, innerSolverMaxItrs=200, 
                 lineSearchMaxItrs=50, gradTol=1e-10, beta=1e-4, 
                 beta2=1e-1, epsilon=1e-6, show=True, record_txt=None):
    iters = 0
    orcl = 0
    # x = copy.deepcopy(x0)    
    x = x0.clone()
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()        
    alphak = 1
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    dType = 'None'
    
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, 1], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS, iterSolver, 
                    rel_res, dType)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    

        t0 = time()
        # p, rel_res, iterSolver, dType = modifiedCR(
        #     Hk, -gk, innerSolverMaxItrs, epsilon, min(innerSolverTol,np.sqrt(gk)))  
        p, rel_res, iterSolver, dType = modifiedCR(
            Hk, -gk, innerSolverMaxItrs, epsilon, innerSolverTol) 
        if torch.isnan(p).any():
            break
            
        x, alphak, iterLS, iterLS_orcl = linesearchzoom(
            obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, 
            c1=beta, c2=beta2, fe=funcEvalMax-orcl)
        orcl += orc_call(iterSolver, HProp, iterLS_orcl)
        
        if alphak < is_zero:
            orcl = funcEvalMax
        else:
            fk, gk, Hk = obj(x)
            gk_norm = gk.norm()
        
        iters += 1  
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, alphak)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('Newton_CR_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record


def Newton_CG_LS(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol2, 
                  innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10, 
                  beta=1e-4, epsilon=0, show=True, record_txt=None, arg=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)    
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()        
    alphak = 1
    alphal = 1
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    dType = None
    rec = False
    # Initialize f0, g0, oracle_call0, time0, alpha0, NC
    record = torch.tensor([fk, gk_norm, 0, 0, 1, 0], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS, iterSolver, rel_res, dType)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    

        t0 = time()
        p, dType, iterSolver, ptHp, rel_res, rec = CappedCG(
                Hk, -gk, innerSolverTol2, 2*epsilon, innerSolverMaxItrs, rec) #to share the same epsilon
        if torch.isnan(p).any():
            break
        if rec:
            print('regenerate')
        else:
            if arg == 'NC_FW':
                if dType == 'NC':
                    pTg = torch.dot(p, gk)
                    # sign_p = - torch.sign(pTg)
                    p = - torch.sign(pTg)*abs(ptHp)*p/torch.norm(p)**3
                    x, alphak, iterLS = linesearch_NewtonCG_Pert_NC(
                            obj, fk, x, p, lineSearchMaxItrs, alphal, c1=beta)
                    alphal = alphak
                else:
                    x, alphak, iterLS = linesearchNC(
                            obj, fk, x, p, lineSearchMaxItrs, 1, c1=beta)
            else:
                if dType == 'NC':
                    pTg = torch.dot(p, gk)
                    # sign_p = - torch.sign(pTg)
                    p = - torch.sign(pTg)*abs(ptHp)*p/torch.norm(p)**3
                x, alphak, iterLS = linesearchNC(
                        obj, fk, x, p, lineSearchMaxItrs, 1, c1=beta)
            if alphak < is_zero:
                orcl = funcEvalMax
            else:
                fk, gk, Hk = obj(x)
                gk_norm = gk.norm()
            # print(dType)
            orcl += orc_call(iterSolver, HProp, iterLS)
                
            iters += 1  
            tmk += time()-t0
                    
            record = recording(record, fk, gk_norm, orcl, tmk, alphak, dType)
            if record_txt is not None and orcl >= orcl_sh:
                record_txt('Newton_CG_pert_%s' % orcl_sh, record)
                orcl_sh = orcl_sh*10
    return x, record


def Newton_CG_TR_Steihaug(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
                 innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10, 
                 Delta=1E5, Delta_max=1E10, 
                 gama1 = 2, gama2 = 0.5, eta = 0.75, 
                 show=True, record_txt=None):
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)    
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()   
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    
    # Initialize f0, g0, oracle_call0, time0, alpha0
    record = torch.tensor([fk, gk_norm, 0, 0, Delta], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, Delta, iterLS, iterSolver, 
                    rel_res)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    

        t0 = time()
        p, rel_res, iterSolver, mp, dType, dNorm, flag =  SteihaugCG(
            Hk, -gk, innerSolverTol, innerSolverMaxItrs, Delta)

        orcl += orc_call(iterSolver, HProp) + 2 # 2 for computing mp 
            
        if mp > -is_zero:
            Delta = 0
            p = torch.zeros_like(x)
            
        iters += 1  
        fkl, gkl, Hkl = fk, gk, Hk
        xl = x
        x = xl + p
        fk, gk, Hk = obj(x)
        gk_norm = gk.norm()
        rho = - (fkl - fk)/mp
        # print('rho', rho, p.norm(), fkl, fk)
        
        if eta >= 1/4:
            print('Warning, Trust-Region Newton-CG eta too large')
        if rho < 1/4:
            Delta = Delta*gama2
        else:
            if rho > 3/4 and dNorm == 'Equal':
                Delta = min(Delta*gama1, Delta_max)
        # print('Delta', Delta, rho, mp, rel_res, iterSolver, flag)
        if rho < eta:
            x = xl
            fk, gk, Hk = fkl, gkl, Hkl
#        else:
#            print('successful', Delta)
        if Delta < is_zero:
            orcl = funcEvalMax
        
        tmk += time()-t0

        record = recording(record, fk, gk_norm, orcl, tmk, Delta)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('Newton_CG_TR_Steihaug_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record

def Newton_CG_TR_Pert(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
                       innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
                       epsilon=1e-5, Delta=1E5, Delta_max=1E10, 
                       gama1 = 2, gama2 = 0.5, eta = 0.75, 
                       show=True, record_txt=None):
    
    iters = 0
    orcl = 0
    x = copy.deepcopy(x0)    
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()   
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    dType = None
    
    # Initialize f0, g0, oracle_call0, time0, Delta, NC
    record = torch.tensor([fk, gk_norm, 0, 0, Delta, 0], device=x.device, dtype=torch.float64).reshape(1,-1)
    orcl_sh = orcl_every_record
    while True:
        if (show and iters%num_every_print == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, Delta, iterLS, iterSolver, 
                    rel_res, dType)
        
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break    

        t0 = time()
        p, rel_res, iterSolver, mp, dType =  myCG_TR_Pert(
            Hk, -gk, innerSolverTol, innerSolverMaxItrs, Delta, 
            epsilon)
        
        orcl += orc_call(iterSolver, HProp) + 2 
            
        if mp > -is_zero:
            Delta = 0
            p = torch.zeros_like(x)
            
        iters += 1  
        fkl, gkl, Hkl = fk, gk, Hk
        xl = x
        x = xl + p
        fk, gk, Hk = obj(x)
        gk_norm = gk.norm()
        rho = - (fkl - fk)/mp
        if rho >= eta:
            Delta = min(gama1*Delta, Delta_max)
#            print('successful', Delta)
        else:
            Delta = gama2*Delta
            x = xl
            fk, gk, Hk = fkl, gkl, Hkl
        if Delta < is_zero:
            orcl = funcEvalMax
        
        tmk += time()-t0
                
        record = recording(record, fk, gk_norm, orcl, tmk, Delta, dType)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('Newton_CG_TR_pert_%s' % orcl_sh, record)
            orcl_sh = orcl_sh*10
    return x, record


