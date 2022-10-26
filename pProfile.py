import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def pProfile(methods, Matrix, length=300, arg='log10', cut=None, 
             xlabel=r'$\tau$', ylim=None, Input_positive=True, leg=False):
    """
    A performance profile plot is the CDF of the performance of every 
    algorithms. Put your pProfile record (objVal/gradNorm/err) matrix to 
    pProfile folder and run this file to regenerate plots with proper scale.
    
    Reference:        
        E. D. Dolan and J. J. More. Benchmarking optimization software with 
        performance pro
files. Mathematical programming, 91(2): 201-213, 2002.
        
        N. Gould and J. Scott. A note on performance pro
les for benchmarking 
        software. ACM Transactions on Mathematical Software (TOMS), 43(2):15, 
        2016.
    INPUT:
        methods: list of all methods
        Matrix: a record matrix s.t., every row are the best (f/g/error) result 
            of all listed methods. e.g., 500 total_run of 5 different 
            optimisation methods will be a 500 x 5 matrix.
        length: number of nodes to compute on range x
        arg: scalling methods of x-axis
        cute: number of nodes to plot/display
        ylabel: label of y-axis
        ylim: plot/display [ylim, 1]
    OUTPUT:
        performance profile plots of the cooresponding (f/g/error) record 
        matrix.
    """
    M_hat = np.min(Matrix, axis=1)
    if not Input_positive:
        ### If M_hat contains negative number, 
        ### shift all result with - 2*M_hat, Then the minimum = -M_hat 
        ### and all other results are larger accordingly 
        Matrix[np.where(M_hat<0)] = (Matrix[np.where(M_hat<0)].T - M_hat[np.where(M_hat<0)]*2).T
        M_hat[np.where(M_hat<0)] = -M_hat[np.where(M_hat<0)]
    R = Matrix.T/M_hat
    R = R.T   
    x_high = np.max(R)
    ftsize = 12
    # x_high = 1E45
    # index = np.max(R,1) <= x_high
    # n2 = sum(index)
    # R = R[index]
    if arg is None:
        x = np.linspace(1, x_high, length)
        myplt = plt.plot
        plt.xlabel(xlabel)
    if arg == 'log2':
        x = np.logspace(0, np.log2(x_high), length)
        myplt = plt.semilogx
        plt.xlabel(r'log2$(\tau)$')
    if arg == 'log10':
        x = np.logspace(0, np.log10(x_high), length)
        myplt = plt.semilogx
        plt.xlabel(xlabel)
        # plt.xlabel(r'log$(\tau)$')
    n, d = Matrix.shape
    if cut != None:
        x = x[:cut]
        length = cut
    
    # colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    # linestyles = ['--', '-.', '-']
    # colors = [ 'g', 'b', 'y', 'r', 'm', 'k', 'm', 'b', 'g']
    # linestyles = ['--', '-', '--', '-', '--', '-', '-.', '-.', '-.', '-.']
    colors =      [ 'g', 'b', 'y', 'r', 'm', 'c',  'k', 'b', 'g']
    linestyles = ['--', '--', '-.', '-.', '-', '-.', '-', '-', '-.', '-.']
    
    if ylim != None:
        axes = plt.axes()
        axes.set_ylim(ylim, 1)
    
    for i in range(len(methods)):
        myMethod = methods[i]
        print(myMethod)
#        loop_i = int((i+1)/7)
        Ri = np.tile(R[:,i], (length, 1))
        xi = np.tile(x, (n,1))
        yi = np.sum((Ri.T <= xi), axis=0)/n
        myplt(x, yi, color=colors[i], linestyle=linestyles[i], 
              label = myMethod)
        # plt.plot(np.log(x), yi, color=colors[i], linestyle=linestyles[i], 
        #       label = myMethod)
        
    # plt.xlabel('$\lambda$')
    plt.ylabel(r'Pr$(ratio \leq \tau)$')
    if leg:
        # plt.legend()
        plt.legend(fontsize=ftsize)
    
    
def main():        
    with open('pProfile/methods.txt', 'r') as myfile:
        methods = myfile.read().split()
    F = np.loadtxt(open("pProfile/rand/1/objVal.txt","rb"),delimiter=",",skiprows=0)
    F2 = np.loadtxt(open("pProfile/rand/2/objVal.txt","rb"),delimiter=",",skiprows=0)
    F = np.append(F, F2, axis=0)
    F3 = np.loadtxt(open("pProfile/rand/3/objVal.txt","rb"),delimiter=",",skiprows=0)
    F = np.append(F, F3, axis=0)
    F4 = np.loadtxt(open("pProfile/rand/4/objVal.txt","rb"),delimiter=",",skiprows=0)
    F = np.append(F, F4, axis=0)
    # F5 = np.loadtxt(open("pProfile/randn/5/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F5, axis=0)
    # F = np.loadtxt(open("pProfile/randn/1/objVal2.txt","rb"),delimiter=",",skiprows=0)
    # F2 = np.loadtxt(open("pProfile/randn/2/objVal2.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F2, axis=0)
    # F3 = np.loadtxt(open("pProfile/randn/3/objVal2.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F3, axis=0)
    # F4 = np.loadtxt(open("pProfile/randn/4/objVal2.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F4, axis=0)
    # F5 = np.loadtxt(open("pProfile/randn/5/objVal2.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F5, axis=0)
    # F = np.loadtxt(open("pProfile/randn/1/objVal3.txt","rb"),delimiter=",",skiprows=0)
    # F2 = np.loadtxt(open("pProfile/randn/2/objVal3.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F2, axis=0)
    # F3 = np.loadtxt(open("pProfile/randn/3/objVal3.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F3, axis=0)
    # F4 = np.loadtxt(open("pProfile/randn/4/objVal3.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F4, axis=0)
    # F5 = np.loadtxt(open("pProfile/randn/5/objVal3.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F5, axis=0)
    # F = np.loadtxt(open("pProfile/randn/1/objVal4.txt","rb"),delimiter=",",skiprows=0)
    # F2 = np.loadtxt(open("pProfile/randn/2/objVal4.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F2, axis=0)
    # F3 = np.loadtxt(open("pProfile/randn/3/objVal4.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F3, axis=0)
    # F4 = np.loadtxt(open("pProfile/randn/4/objVal4.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F4, axis=0)
    # F5 = np.loadtxt(open("pProfile/randn/5/objVal4.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F5, axis=0)
    # isnan = np.argwhere(np.isnan(F))
    # a, b = isnan.shape
    # for i in range(a):
    #     F[isnan[i][0], isnan[i][1]] = 10*max(F[isnan[i][0]])
    
    G = np.loadtxt(open("pProfile/rand/1/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    G2 = np.loadtxt(open("pProfile/rand/2/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    G = np.append(G, G2, axis=0)
    G3 = np.loadtxt(open("pProfile/rand/3/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    G = np.append(G, G3, axis=0)
    G4 = np.loadtxt(open("pProfile/rand/4/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    G = np.append(G, G4, axis=0)
    # G5 = np.loadtxt(open("pProfile/randn/5/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G5, axis=0)
    # G = np.loadtxt(open("pProfile/randn/1/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
    # G2 = np.loadtxt(open("pProfile/randn/2/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G2, axis=0)
    # G3 = np.loadtxt(open("pProfile/randn/3/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G3, axis=0)
    # G4 = np.loadtxt(open("pProfile/randn/4/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G4, axis=0)
    # G5 = np.loadtxt(open("pProfile/randn/5/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G5, axis=0)
    # G = np.loadtxt(open("pProfile/randn/1/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
    # G2 = np.loadtxt(open("pProfile/randn/2/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G2, axis=0)
    # G3 = np.loadtxt(open("pProfile/randn/3/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G3, axis=0)
    # G4 = np.loadtxt(open("pProfile/randn/4/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G4, axis=0)
    # G5 = np.loadtxt(open("pProfile/randn/5/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G5, axis=0)
    # isnan = np.argwhere(np.isnan(G))
    # a, b = isnan.shape
    # for i in range(a):
    #     G[isnan[i][0], isnan[i][1]] = 10*max(G[isnan[i][0]])
    np.savetxt(os.path.join('pProfile', 'objVal.txt'), F, delimiter=',')
    np.savetxt(os.path.join('pProfile', 'gradNorm.txt'), G, delimiter=',')
    FF = torch.tensor(F)
    GG = torch.tensor(G)
    # index_F = torch.min(FF, 1)[0] == 0
    # nb = sum(index_F)
    n, m = F.shape
    for i in range(n):
        if torch.min(FF[i,:]) == 0:    
            if torch.max(FF[i,:]) == 0:
                FF[i,:] = FF[i,:] + 1
            else:
                adj = torch.min(FF[i,:][FF[i,:]!=0])*1E-5
                FF[i,:] = FF[i,:] + adj.item()
                
        if torch.min(GG[i,:]) == 0:
            if torch.max(GG[i,:]) == 0:
                GG[i,:] = GG[i,:] + 1
            else:
                adj = torch.min(GG[i,:][GG[i,:]!=0])*1E-5
                GG[i,:] = GG[i,:] + adj.item()
    mypath = 'pProfile'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    figsz = (8,5)
    mydpi = 600
    length = 300
    # myorder = [0,1,2,3,4,5,6]
    myorder = [1, 2, 5, 6, 4, 3, 0]
    methodss = methods.copy()
    for j in range(len(myorder)):
        methodss[j] = methods[myorder[j]]
        F[:,j] = FF[:,myorder[j]]
        G[:,j] = GG[:,myorder[j]]
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methodss, F, length, arg='log10',cut=80, Input_positive=False, leg=True) #80 150
    fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)
    pProfile(methodss, G, length, arg='log10',cut=80, Input_positive=True)
    fig2.savefig(os.path.join(mypath, 'GradientNorm'), dpi=mydpi)
    
# =============================================================================
#     with open('pProfile/methods.txt', 'r') as myfile:
#         methods = myfile.read().split()
#     F = np.loadtxt(open("pProfile/objVal1.txt","rb"),delimiter=",",skiprows=0)
#     F2 = np.loadtxt(open("pProfile/objVal2.txt","rb"),delimiter=",",skiprows=0)
#     F3 = np.loadtxt(open("pProfile/objVal3.txt","rb"),delimiter=",",skiprows=0)
#     F4 = np.loadtxt(open("pProfile/objVal4.txt","rb"),delimiter=",",skiprows=0)
#     # F5 = np.loadtxt(open("pProfile/objVal5.txt","rb"),delimiter=",",skiprows=0)
#     # F6 = np.loadtxt(open("pProfile/objVal6.txt","rb"),delimiter=",",skiprows=0)
#     F = np.append(F, F2, axis=0)
#     F = np.append(F, F3, axis=0)
#     F = np.append(F, F4, axis=0)
#     # F = np.append(F, F5, axis=0)
#     # F = np.append(F, F6, axis=0)
#     G = np.loadtxt(open("pProfile/gradNorm1.txt","rb"),delimiter=",",skiprows=0)
#     G2 = np.loadtxt(open("pProfile/gradNorm2.txt","rb"),delimiter=",",skiprows=0)
#     G3 = np.loadtxt(open("pProfile/gradNorm3.txt","rb"),delimiter=",",skiprows=0)
#     G4 = np.loadtxt(open("pProfile/gradNorm4.txt","rb"),delimiter=",",skiprows=0)
#     # G5 = np.loadtxt(open("pProfile/gradNorm5.txt","rb"),delimiter=",",skiprows=0)
#     # G6 = np.loadtxt(open("pProfile/gradNorm6.txt","rb"),delimiter=",",skiprows=0)
#     G = np.append(G, G2, axis=0)
#     G = np.append(G, G3, axis=0)
#     G = np.append(G, G4, axis=0)
#     # G = np.append(G, G5, axis=0)
#     # G = np.append(G, G6, axis=0)
#     np.savetxt(os.path.join('pProfile', 'objVal.txt'), F, delimiter=',')
#     np.savetxt(os.path.join('pProfile', 'gradNorm.txt'), G, delimiter=',')
#     F[F==0] = 1e-30
#     G[G==0] = 1e-30
# #    Err = np.loadtxt(open("pProfile/err.txt","rb"),delimiter=",",skiprows=0)
#     
#     mypath = 'pProfile'
#     if not os.path.isdir(mypath):
#        os.makedirs(mypath)
#     figsz = (16,12)
#     mydpi = 200
#     length = 300
#     
#     fig1 = plt.figure(figsize=figsz)    
#     pProfile(methods, F, length, arg='log10',cut=300, ylabel='Performance Profile on f', Input_positive=False) #80 150
#     fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
#     
#     fig2 = plt.figure(figsize=figsz)    
#     pProfile(methods, G, length, arg='log10',cut=300, ylabel='Performance Profile on g')
#     fig2.savefig(os.path.join(mypath, 'GradientNorm'), dpi=mydpi)
# =============================================================================
    
#    fig3 = plt.figure(figsize=figsz)    
#    pProfile(methods, Err,length, arg='log10',cut=40, ylabel='Error') #40 80
#    fig3.savefig(os.path.join(mypath, 'Error'), dpi=mydpi)
    
if __name__ == '__main__':
    main()