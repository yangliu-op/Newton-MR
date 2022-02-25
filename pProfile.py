import numpy as np
import os
import matplotlib.pyplot as plt

def pProfile(methods, Matrix, length=300, arg='log10', cut=None, 
             ylabel='Performance Profile', ylim=None, Input_positive=True):
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
    
    M_hat = np.min(Matrix, axis=1) + 1E-50
    if not Input_positive:
        ### If M_hat contains negative number, 
        ### shift all result with - 2*M_hat, Then the minimum = -M_hat 
        ### and all other results are larger accordingly 
        Matrix[np.where(M_hat<0)] = (Matrix[np.where(M_hat<0)].T - M_hat[np.where(M_hat<0)]*2).T
        M_hat[np.where(M_hat<0)] = -M_hat[np.where(M_hat<0)]
    R = Matrix.T/M_hat
    R = R.T   
    x_high = np.max(R)
    if arg is None:
        x = np.linspace(1, x_high, length)
        myplt = plt.plot
        plt.xlabel('lambda')
    if arg == 'log2':
        x = np.logspace(0, np.log2(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log2(lambda)')
    if arg == 'log10':
        x = np.logspace(0, np.log10(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log(lambda)')
    n, d = Matrix.shape
    if cut != None:
        x = x[:cut]
        length = cut
    
    # colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    # linestyles = ['--', '-.', '-']
    colors = ['k', 'm', 'b', 'g', 'y', 'r', 'k', 'm', 'b', 'g']
    linestyles = ['-', '-', '-', '--', '--', '--', '-.', '-.', '-.', '-.']
    
    if ylim != None:
        axes = plt.axes()
        axes.set_ylim(ylim, 1)
    
    for i in range(len(methods)):
        myMethod = methods[i]
#        loop_i = int((i+1)/7)
        Ri = np.tile(R[:,i], (length, 1))
        xi = np.tile(x, (n,1))
        yi = np.sum((Ri.T <= xi), axis=0)/n
        # myplt(x, yi, color=colors[i-7*loop_i], linestyle=linestyles[loop_i], 
        #       label = myMethod)
        myplt(x, yi, color=colors[i], linestyle=linestyles[i], 
              label = myMethod)
        
    plt.ylabel(ylabel)
    plt.legend()
    
    
def main():        
    with open('pProfile/methods.txt', 'r') as myfile:
        methods = myfile.read().split()
    F = np.loadtxt(open("pProfile/ALL_1E5/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F2 = np.loadtxt(open("pProfile/TR_1E5_10_100/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F2, axis=0)
    # F3 = np.loadtxt(open("pProfile/TR_1E5_100/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F3, axis=0)
    # F4 = np.loadtxt(open("pProfile/RandLS_10/LS4/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F4, axis=0)
    # F5 = np.loadtxt(open("pProfile/LS_10/LS5/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F5, axis=0)
    # F6 = np.loadtxt(open("pProfile/LS_10/LS6/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F6, axis=0)
    # F7 = np.loadtxt(open("pProfile/LS_10/LS7/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F7, axis=0)
    # F8 = np.loadtxt(open("pProfile/LS_100/LS8/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F8, axis=0)
    # F9 = np.loadtxt(open("pProfile/LS_100/LS9/objVal.txt","rb"),delimiter=",",skiprows=0)
    # F = np.append(F, F9, axis=0)
    
    G = np.loadtxt(open("pProfile/ALL_1E5/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G2 = np.loadtxt(open("pProfile/TR_1E5_10_100/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G2, axis=0)
    # G3 = np.loadtxt(open("pProfile/TR_1E5_100/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G3, axis=0)
    # G4 = np.loadtxt(open("pProfile/RandLS_10/LS4/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G4, axis=0)
    # G5 = np.loadtxt(open("pProfile/LS_10/LS5/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G5, axis=0)
    # G6 = np.loadtxt(open("pProfile/LS_10/LS6/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G6, axis=0)
    # G7 = np.loadtxt(open("pProfile/LS_10/LS7/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G7, axis=0)
    # G8 = np.loadtxt(open("pProfile/LS_100/LS8/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G8, axis=0)
    # G9 = np.loadtxt(open("pProfile/LS_100/LS9/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    # G = np.append(G, G9, axis=0)
    np.savetxt(os.path.join('pProfile', 'objVal.txt'), F, delimiter=',')
    np.savetxt(os.path.join('pProfile', 'gradNorm.txt'), G, delimiter=',')
    F[F==0] = 1e-30
    G[G==0] = 1e-30
#    Err = np.loadtxt(open("pProfile/err.txt","rb"),delimiter=",",skiprows=0)
    
    mypath = 'pProfile'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    figsz = (16,12)
    mydpi = 200
    length = 300
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methods, F, length, arg='log10',cut=80, ylabel='Performance Profile on f', Input_positive=False) #80 150
    fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)    
    pProfile(methods, G, length, arg='log10',cut=80, ylabel='Performance Profile on g')
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