import matplotlib.pyplot as plt
import os
import torch
import copy
from optim_algo import recording
from myMINRES import Ax, symGivens
import matplotlib.ticker as ticker
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

global colors, linestyles, markers
colors = [ 'k', 'b', 'm',  'c','g', 'r', 'y']
#colors = [(0,0,0), (230/255,0,128/255), (0,128/255,1), (128/255,230/255,0), (1,0,0), 'b', 'c']
linestyles = ['--', '-',  '--', '-',   '-.', ':', ':', ':']
markers = ['o', '*', 'x', '1', '2', '+', 'o']
markers1 = ['o', 'X', '*', 'P', '1', '2', '.']
linewidth1 = [1.8, 1.8, 1.8, 1.8]
markersize1 = [3, 3, 4, 3]
def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None, index=None):
    # record = record
    # record = myRound(record)
    if not (xaxis is not None):
        xaxis = torch.tensor(range(0,len(yaxis)))
    plt_fun(xaxis, yaxis, color=colors[i], linewidth=linewidth1[i], linestyle=linestyles[i], 
            marker=markers1[i], markersize=markersize1[i], label = label)
    if NC:
        if index is None:
            index = (record[:,5] == True)
#            xNC = xaxis[index]
##            if xaxis is not None:
##                xNC = xaxis[:-1][index]
##            else:
##                xNC = torch.tensor(range(1,len(yaxis)))[index]
#            yNC = yaxis[index]
#        else:
        xNC = xaxis[index]
        yNC = yaxis[index]
        plt_fun(xNC, yNC, '.', color=colors[i], 
                marker=markers1[i], markersize=markersize1[i]*2.4)
        
        
def myRound(x, dec=2):
    tmp = 10**dec
    return (x*tmp).round()/tmp

def showFigure_MINRES(methods_all, record_all, mypath):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Iteration vs. Step Size
    """
    fsize = 14
    # myplt = plt.loglog
#    myplt = plt.semilogx
    myplt = plt.plot
#    myplt = plt.scatter
    myplt2 = plt.semilogy
    
    figsz = (20,10)
    mydpi = 300
    
    # record = recording(record, mk, bTx, xnorm, minT, relres, dType)
    fig1 = plt.figure(figsize=figsz)
    
    plt.subplot(231)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        yaxis = record[1:,3]
        xaxis = torch.tensor(range(1,len(yaxis)+1))
#        myplt(xaxis, yaxis, color=colors[i], linestyle=linestyles[0], 
#              label = methods_all[i])
        index = (record[:,5][1:] == True)
#        xNC = xaxis[index]
#        yNC = yaxis[index]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), 
             yaxis, xaxis=xaxis, index=index)
#        myplt(xNC, yNC, '.', color=colors[i], 
#                marker=markers[i], markersize=8)
    # plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel(r'$ \lambda_{\min}(T_k) $', fontsize=fsize)
    plt.yscale('symlog')
    plt.grid(True)
    plt.legend()
            
    plt.subplot(232)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), -2*record[:,0]-record[:,1] )
#    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel(r'$ \left\langle x_k, r_k \right\rangle $', fontsize=fsize)
    # plt.yscale('symlog')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    plt.grid(True)
    plt.legend()
    
    plt.subplot(233)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0])
#    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel(r'$ m(x_k) $', fontsize=fsize)
    # plt.yscale('symlog')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    plt.grid(True)
    plt.legend()
            
    plt.subplot(234)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,2])
#    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel(r'$ || x_k || $', fontsize=fsize)
    # plt.yscale('symlog')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    plt.grid(True)
    plt.legend()
    
#    fig5 = plt.figure(figsize=figsz)
#    for i in range(len(methods_all)):
#        record = copy.deepcopy(record_all[i])
#        yaxis = record[1:,2]
#        xaxis = torch.tensor(range(1,len(yaxis)+1))
##        myplt2(xaxis, yaxis, color=colors[i], linestyle=linestyles[0], 
##              label = methods_all[i])
#        index = (record[:,5][1:] == True)
##        xNC = xaxis[index]
##        yNC = yaxis[index]
##        myplt2(xNC, yNC, '.', color=colors[i], 
##                marker=markers[i], markersize=8)
#        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), 
#             yaxis, xaxis=xaxis, index=index)
#    plt.xlabel('Iteration', fontsize=fsize)
#    plt.ylabel(r'$ \| x_t \| $', fontsize=fsize)
#    plt.grid(True)
#    plt.legend()
#    fig5.savefig(os.path.join(mypath, 'xnorm'), dpi=mydpi)
#    fig5.savefig(os.path.join(mypath, 'xnorm.eps'), format='eps', dpi=2*mydpi)
    
    plt.subplot(235)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1])
#    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel(r'$ \left\langle x_k, b \right\rangle $', fontsize=fsize)
    # plt.yscale('symlog')
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    plt.grid(True)
    plt.legend()
    
    plt.subplot(236)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        draw(myplt2, record, methods_all[i], i, (record.shape[1]==6), record[:,4])
#    plt.xlabel('Iteration', fontsize=fsize)
    plt.ylabel('Relres', fontsize=fsize)
    plt.grid(True)
    plt.legend()
    fig1.savefig(os.path.join(mypath, 'minres'), dpi=mydpi)
    
#    fig1.savefig(os.path.join(mypath, 'm'), dpi=mydpi)
#    fig2.savefig(os.path.join(mypath, 'relres'), dpi=mydpi)
#    fig3.savefig(os.path.join(mypath, 'bTx'), dpi=mydpi)
#    fig4.savefig(os.path.join(mypath, 'rTx'), dpi=mydpi)
#    fig5.savefig(os.path.join(mypath, 'xnorm'), dpi=mydpi)
#    fig6.savefig(os.path.join(mypath, 'lambda'), dpi=mydpi)
    
#    fig1.savefig(os.path.join(mypath, 'm.eps'), format='eps', dpi=2*mydpi)
#    fig2.savefig(os.path.join(mypath, 'Relres.eps'), format='eps', dpi=2*mydpi)
#    fig3.savefig(os.path.join(mypath, 'bTx.eps'), format='eps', dpi=2*mydpi)
#    fig4.savefig(os.path.join(mypath, 'rTx.eps'), format='eps', dpi=2*mydpi)
#    fig5.savefig(os.path.join(mypath, 'xnorm.eps'), format='eps', dpi=2*mydpi)
#    fig6.savefig(os.path.join(mypath, 'lambda.eps'), format='eps', dpi=2*mydpi)
        
    
def MINRES_exp(A, b, rtol, maxit, shift=0,
               reOrth=True, isZero=1E-8):
    
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    dltan = 0
    eplnn = 0
    gama = 0
#    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    dType = 0
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 9        
        
    # if b.T.dot(Ax(A, b)) <= 0: # covered
    #     flag = 3
    #     #print('ncb', -nc)
    #     return b, relres, iters
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
#    normHy2 = tau**2
    record = torch.tensor([0, 0, 0, 0, 1, 0], device=x.device, dtype=torch.float64).reshape(1,-1)
    while flag == flag0:
        #lanczos
        betal = beta
        beta = betan
        v = vn
#        print(v)
        r3 = Ax(A, v)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
        
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
                    print('flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
                    print('flag = -1')
                    break
                
            
        vn = r3/betan
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
            V = Vn[:,:-1]
            if iters == 1:
                alpha_diag = alfa.reshape(1)
                beta_diag = betan .reshape(1) 
                T = alfa
                minT = alfa
                bT = torch.cat((T.reshape(1), betan.reshape(1)), 0).reshape(2, 1)
            else:
                alpha_diag = torch.cat((alpha_diag, alfa.reshape(1)), axis=0)
                beta_diag = torch.cat((beta_diag, betan.reshape(1)), axis=0)   
                T = torch.diag(beta_diag[:-1],-1) + torch.diag(
                        alpha_diag,0) + torch.diag(beta_diag[:-1],1)   
                minT = min(torch.eig(T)[0][:,0])
                # t1 = torch.zeros(1,iters,device=b.device, dtype=torch.float64)
                # t1[0,-1] = betan
                # bT = torch.cat((T, t1), axis=0)  
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
        
        nc = csl * gbar 
        
        if nc >= -1E-14:
#            flag = 3
            print('NC')
            dType = 'NC'
        else:            
            dType = 'Sol'
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan) 
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            rk = sn**2 * rkl - phi * cs * vn
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
#            gama = 0  
            phi = phil
            rk = rkl
            x = xl
            flag = 2
            print('flag = 2, b is not in the range(A)!')
            maxit += 1
            
        ## stopping criteria
        xnorm = torch.norm(x)   
        rnorm = phi
        relres = rnorm / beta1
        bTx = b @ x
        mk = -rk @ x/2 - bTx/2
        xnorm = x.norm()
        print(iters, mk, bTx, xnorm, minT, relres, gama)
        
        record = recording(record, mk, bTx, xnorm, minT, relres, dType)
        
        if relres <= rtol:
            return x, record
        if iters > maxit:
            flag = 4  ## exit before maxit
            # print('Maximun iteration reached', flag, iters)
            return x, record
    return x, record

# def main():
#     methods_all = []
#     record_all = []
# #    for method in os.listdir('showFig'): #only contains txt files
# #        methods_all.append(method.rsplit('.', 1)[0])
# #        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
# #        record_all.append(record)
#     for method in os.listdir('showFig'): #only contains txt files
#         methods_all.append(method.rsplit('.', 1)[0])
#         print(method)
#         record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
#         record = record[:81,:]
#         if method == 'AndersonAcc_general.txt':
#             print((record[:,1][record[:,1]>1E-7]).shape)
#             print(record[:,1][(record[:,1][record[:,1]>1E-7]).shape])
#         record_all.append(record)
#     mypath = 'showFig_plots'
#     if not os.path.isdir(mypath):
#        os.makedirs(mypath)
# # =============================================================================
#     """
#     regenerate any 1 total run plot via txt record matrix in showFig folder.
#     Note that directory only contains txt files.
#     For performace profile plots, see pProfile.py
#     """
#     showFigure(methods_all, record_all, None, mypath)
    
def generate_A(n, d, eigV, device):    
    A = torch.randn(n, n, device=device, dtype=torch.float64)
    D = A + A.T
    U, s, V = torch.svd(D)
    # ss = abs(-1,2,n)
    A = U @ torch.diag(eigV) @ U.T
    return A
    
def main(): 
    """Run an example of minresQLP."""
    curr_seed = 2
    torch.manual_seed(curr_seed)
    device = torch.device("cpu")
    eigMax = 3
    eigMin = 0
    n = 20
    d = n
    b = torch.ones(n, device=device, dtype=torch.float64)
    eigV1 = torch.zeros(n, device=device, dtype=torch.float64)
    sp = torch.logspace(eigMax,eigMin,d-1, dtype=torch.float64)
    print(sp)
    eigV1[:d-1] = sp
#    eigV1[d-1] = sm
    A = generate_A(n, d, eigV1, device)  
    
    # d = 20
    b = torch.ones(n, device=device, dtype=torch.float64)
    eigV2 = torch.zeros(n, device=device, dtype=torch.float64)
    sp = torch.logspace(eigMax,eigMin,d-1, dtype=torch.float64)
    sm = -1E0
#    sm = -sm*10
    eigV2[:d-1] = sp
    eigV2[d-1] = sm
    B = generate_A(n, d, eigV2, device)   
    
    # d = 20
    dm = 1
    b = torch.ones(n, device=device, dtype=torch.float64)
    eigV3 = torch.zeros(n, device=device, dtype=torch.float64)
    sp = torch.logspace(eigMax,eigMin,d-1, dtype=torch.float64)
    # sm = -torch.logspace(-3,-1,5, dtype=torch.float64)
    # sm = -1E-3
    eigV3[:d-dm] = sp[:d-dm]
    eigV3[-2] = sm
    eigV3[-1] = sm*10
    C = generate_A(n, d, eigV3, device)   
    myMethod = ['A',
               'B',
               'C',
        ]
    matrix = [A,
               B,
               C,
        ]
    record_all = []  
    for i in range(len(matrix)):
        record_all.append(myMethod[i])
        x, record = MINRES_exp(matrix[i], b, 0, d-1)
        record_all.append(record)
    
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    mypath = 'MINRES_n_%s_seed_%s' % (
                    n, curr_seed)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    showFigure_MINRES(methods_all, record_all, mypath)
    
    
if __name__ == '__main__':
    main()