import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
global colors, linestyles, markers
# colors = [(0,0,0), (230/255,0,128/255), (128/255,230/255,0), (0,128/255,1), (1,0,0), 'b', 'c']
colors = [ 'g', 'b', 'y', 'r', 'm', 'c', 'k', 'b', 'g', 'b', 'y', 'r']
linestyles = ['--', '--', '-.', '-.', '-', '--', '-', '-.', '-.', '-.']
# colors = ['m', 'g', 'y', 'r', 'b', 'k', 'c']
# linestyles = [ '-', '-', '-', '-', '--',  '--',  '--',  '-.', '-.', ':', ':', '-', ':']
markers = ['*',  '1', '2', 'o', '+', 'o', '*',  '1', '2', 'o', '+', 'o']
markers1 = ['P', 'X', '2', 'P',  '.','*', 'x', 'X', '2', 'P',  '.','*', 'x']
linewidth1 = [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
markersize1 = [5, 3, 5, 5, 5, 6, 5, 3, 5, 5, 5, 6, 5]

def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None, resulted=True):
    # loop_i = int((i+1)/5)
    if not (xaxis is not None):
        xaxis = torch.tensor(range(1,len(yaxis)+1))
    plt_fun(xaxis, yaxis, color=colors[i], linewidth=linewidth1[0], linestyle=linestyles[i], label = label)
    # plt_fun(xaxis, yaxis, color=colors[i-7*loop_i], 
    #         linestyle=linestyles[loop_i], label = label)
    if NC:
        if resulted:
            index = (record[:,5][:-1] == True)
            if xaxis is not None:
                xNC = xaxis[:-1][index]
            else:
                xNC = torch.tensor(range(1,len(yaxis)))[index]
            yNC = yaxis[:-1][index]
        else:
            index = (record[:,5][1:] == True)
            if xaxis is not None:
                xNC = xaxis[:-1][index]
            else:
                xNC = torch.tensor(range(1,len(yaxis)))[index]
            yNC = yaxis[:-1][index]
        plt_fun(xNC, yNC, '.', color=colors[i], 
                marker=markers1[i], markersize=markersize1[i])
        
def showFigure(methods_all, record_all, prob, mypath, plotAll=False):
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
    fsize = 20
    ftsize = 9
    myplt = plt.loglog
#    myplt = plt.semilogx
#    myplt = plt.semilogy
#    myplt = plt.plot
    
    figsz = (18,10)
    mydpi = 100
    
    fig1 = plt.figure(figsize=figsz)
    Relative_f = False
    # Relative_f = True
    
    if Relative_f:
        F_star = record_all[0][-1,0]
        for i in range(len(methods_all)-1):
            F_star = min(record_all[i+1][-1,0], F_star)

    plt.subplot(221)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        if Relative_f:
            record[:,0] = (record[:,0] - F_star)/max(F_star, 1)
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
            draw(myplt, record, methods_all[i], i, False, record[:,0])
        else:
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    # plt.ylabel(r'$ f $', fontsize=fsize)
    if Relative_f:
        plt.ylabel(r'$\frac{f(x_k) - f^{*}}{\max \{f^{*}, 1\}}$', fontsize=fsize)
    else:
        plt.ylabel(r'$ f $', fontsize=fsize)
    plt.legend(fontsize=ftsize)
    
    
    # fig2 = plt.figure(figsize=figsz)
    plt.subplot(222)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record[:,5] = 0
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1], record[:,2]+1)
    plt.xlabel('Oracle calls', fontsize=fsize)
    if Relative_f:
        plt.ylabel(r'$|| \nabla f(x_k) ||$', fontsize=fsize)
    else:
        plt.ylabel(r'$|| \nabla f ||$', fontsize=fsize)
    
    # plt.legend(fontsize=ftsize)
    # fig2.savefig(os.path.join(mypath, 'fradient_norm'), dpi=mydpi)
    
    plt.subplot(223)
    # fig1 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        if Relative_f:
            record[:,0] = (record[:,0] - F_star)/max(F_star, 1)
        if methods_all[i] == 'GD' or methods_all[i] == 'AndersonAcc_pure':
            record = record[:,:5]
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,3]+1)
    plt.xlabel('Time (s)', fontsize=fsize)  
    if Relative_f:
        plt.ylabel(r'$\frac{f(x_k) - f^{*}}{\max \{f^{*}, 1\}}$', fontsize=fsize)
    else:
        plt.ylabel(r'$ f $', fontsize=fsize)
    # plt.legend(fontsize=ftsize)
    
    
    plt.subplot(224)
    # fig1 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = copy.deepcopy(record_all[i])
        draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,4], record[:,2]+1, resulted=True)
        # draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,4], record[:,2]+1, resulted=True)
    plt.xlabel('Iteration', fontsize=fsize)
    plt.xlabel('Oracle calls', fontsize=fsize)
    plt.ylabel(r'$ \alpha_k $ or $ \Delta_k $', fontsize=fsize)
    # plt.ylabel('Delta', fontsize=fsize)
    # plt.legend(fontsize=ftsize)
    fig1.savefig(os.path.join(mypath, 'normal'), dpi=mydpi)
    # fig4.savefig(os.path.join(mypath, 'delta_alpha'), dpi=mydpi)
            
    if plotAll == True:
        fig4 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = copy.deepcopy(record_all[i])
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig4.savefig(os.path.join(mypath, 'iteration_f'), dpi=mydpi)
        
        fig5 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = copy.deepcopy(record_all[i])
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,1])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig5.savefig(os.path.join(mypath, 'iteration_g'), dpi=mydpi)
        
        fig6 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = copy.deepcopy(record_all[i])
            draw(myplt, record, methods_all[i], i, (record.shape[1]==6), record[:,0], record[:,2]+1)
        plt.xlabel('Oracle calls', fontsize=fsize)
        plt.ylabel('Step Size', fontsize=fsize)
        plt.legend()
        fig6.savefig(os.path.join(mypath, 'oc_alpha'), dpi=mydpi)
        
def main():
    methods_all = []
    record_all = []
    for method in os.listdir('showFig'): #only contains txt files
        methods_all.append(method.rsplit('.', 1)[0])
        print(method)
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        # record = record[:81,:]
        if method == 'AndersonAcc_general.txt':
            print((record[:,1][record[:,1]>1E-7]).shape)
            print(record[:,1][(record[:,1][record[:,1]>1E-7]).shape])
        record_all.append(record)
    mypath = 'showFig_plots'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
# =============================================================================
    """
    regenerate any 1 total run plot via txt record matrix in showFig folder.
    Note that directory only contains txt files.
    For performace profile plots, see pProfile.py
    """
    # myorder = [0, 5, 4, 3, 2, 1, 6]
    # methods_all = [methods_all[i] for i in myorder]
    # record_all = [record_all[i] for i in myorder]
    showFigure(methods_all, record_all, None, mypath)
    
    
    
    
    
    
if __name__ == '__main__':
    main()