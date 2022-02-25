import numpy as np
import torch
from numpy.linalg import inv, svd
from numpy.random import multivariate_normal, randn, rand
from logistic import logit
from optim_algo import (Newton_MR_invex, Newton_CG, L_BFGS, Gauss_Newton, 
                        MomentumSGD, Adagrad, Adadelta, RMSprop, Adam, SGD,
                        Newton_MR, Newton_CR, Damped_Newton_CG, 
                        Newton_MR_TR, Newton_CG_TR, Newton_CG_TR_Pert, 
                        AndersonAcc,
                        )
from loaddata import loaddata
from showFigure import showFigure
from sklearn import preprocessing
from regularizer import regConvex, regNonconvex, L1
from student_t import student_t
import os
# from Auto_Enconder import AE_Call
import matplotlib.pyplot as plt
from pProfile import pProfile
from softmax import softmax
from logitreg import logitreg
from least_square import least_square
from scipy import sparse
#import torch.utils.data as data
import torchvision.datasets as datasets
from  torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
#import os
#print(os.environ)
#os.environ['CUTEST'] = '/home/yang/Documents/cutest/cutest/'
#os.environ['SIFDECODE'] = '/home/yang/Documents/cutest/sifdecode/'
#os.environ['ARCHDEFS'] = '/home/yang/Documents/cutest/archdefs/'
#os.environ['MASTSIF'] = '/home/yang/Documents/cutest/mastsif/'
#os.environ['MYARCH'] = 'pc64.lnx.gf7'
#import pycutest

from torch import nn
import shutil
        
class auto_Encoder_CIFAR10(nn.Module):
    def __init__(self):
        super(auto_Encoder_CIFAR10, self).__init__()
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 2048), ##5
#                                      nn.Sigmoid(),
#                                      nn.Linear(2048, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 128),
#                                      nn.Sigmoid(),
#                                      nn.Linear(128, 32),
#                                      nn.Sigmoid(),
#                                      nn.Linear(32, 8),
#                                      nn.Sigmoid(),
#                                      nn.Linear(8, 3))
#        self.decoder = nn.Sequential(nn.Linear(3, 8),
#                                      nn.Sigmoid(),
#                                      nn.Linear(8, 32),
#                                      nn.Sigmoid(),
#                                      nn.Linear(32, 128),
#                                      nn.Sigmoid(),
#                                      nn.Linear(128, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 2048),
#                                      nn.Sigmoid(),
#                                      nn.Linear(2048, 3*32*32),
#                                      nn.Softplus())
        self.encoder = nn.Sequential(nn.Linear(3*32*32, 509),#L43
                                      nn.Sigmoid(),
                                      nn.Linear(509, 61),
                                      nn.Sigmoid(),
                                      nn.Linear(61, 7))
        self.decoder = nn.Sequential(nn.Linear(7, 61),
                                      nn.Sigmoid(),
                                      nn.Linear(61, 509),
                                      nn.Sigmoid(),
                                      nn.Linear(509, 3*32*32),
                                      nn.Softplus())
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 1021),#L63
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 167),
#                                      nn.Sigmoid(),
#                                      nn.Linear(167, 23),
#                                      nn.Sigmoid(),
#                                      nn.Linear(23, 3))
#        self.decoder = nn.Sequential(nn.Linear(3, 23),
#                                      nn.Sigmoid(),
#                                      nn.Linear(23, 167),
#                                      nn.Sigmoid(),
#                                      nn.Linear(167, 1021),
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 3*32*32),
#                                      nn.Softplus())
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 1021),#L83
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 251),
#                                      nn.Sigmoid(),
#                                      nn.Linear(251, 61),
#                                      nn.Sigmoid(),
#                                      nn.Linear(61, 13),
#                                      nn.Sigmoid(),
#                                      nn.Linear(13, 3))
#        self.decoder = nn.Sequential(nn.Linear(3, 13),
#                                      nn.Sigmoid(),
#                                      nn.Linear(13, 61),
#                                      nn.Sigmoid(),
#                                      nn.Linear(61, 251),
#                                      nn.Sigmoid(),
#                                      nn.Linear(251, 1021),
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 3*32*32),
#                                      nn.Softplus())
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 1021),#L82
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 251),
#                                      nn.Sigmoid(),
#                                      nn.Linear(251, 61),
#                                      nn.Sigmoid(),
#                                      nn.Linear(61, 13),
#                                      nn.Sigmoid(),
#                                      nn.Linear(13, 3))
#        self.decoder = nn.Sequential(nn.Linear(3, 13),
#                                      nn.Sigmoid(),
#                                      nn.Linear(13, 61),
#                                      nn.Sigmoid(),
#                                      nn.Linear(61, 251),
#                                      nn.Sigmoid(),
#                                      nn.Linear(251, 1021),
#                                      nn.Sigmoid(),
#                                      nn.Linear(1021, 3*32*32),
#                                      nn.Softplus())
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 64),
#                                      nn.Sigmoid(),
#                                      nn.Linear(64, 7))
#        self.decoder = nn.Sequential(nn.Linear(7, 64),
#                                      nn.Sigmoid(),
#                                      nn.Linear(64, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 3*32*32),
#                                      nn.Softplus())
#        self.encoder = nn.Sequential(nn.Linear(3*32*32, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 168),
#                                      nn.Sigmoid(),
#                                      nn.Linear(168, 48),
#                                      nn.Sigmoid(),
#                                      nn.Linear(48, 12),
#                                      nn.Sigmoid(),
#                                      nn.Linear(12, 3))
#        self.decoder = nn.Sequential(nn.Linear(3, 12),
#                                      nn.Sigmoid(),
#                                      nn.Linear(12, 48),
#                                      nn.Sigmoid(),
#                                      nn.Linear(48, 168),
#                                      nn.Sigmoid(),
#                                      nn.Linear(168, 512),
#                                      nn.Sigmoid(),
#                                      nn.Linear(512, 3*32*32),
#                                      nn.Softplus())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class auto_Encoder_MNIST(nn.Module):
    def __init__(self):
        super(auto_Encoder_MNIST, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
                                      nn.ReLU(True),
                                      nn.Linear(128, 64),
                                      nn.ReLU(True),
                                      nn.Linear(64, 12),
                                      nn.ReLU(True),
                                      nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                      nn.ReLU(True),
                                      nn.Linear(12, 64),
                                      nn.ReLU(True),
                                      nn.Linear(64, 128),
                                      nn.ReLU(True),
                                      nn.Linear(128, 28*28),
                                      nn.Tanh())
        # self.encoder = nn.Sequential(
        #     nn.Linear(28 * 28, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 12),
        #     nn.ReLU(True))
        # self.decoder = nn.Sequential(
        #     nn.Linear(12, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 28 * 28),
        #     nn.Sigmoid())
        # self.encoder = nn.Sequential(
        #     nn.Linear(300, 28),
        #     nn.Sigmoid(),
        #     nn.Linear(28, 3),
        #     nn.Sigmoid())
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 28),
        #     nn.Sigmoid(),
        #     nn.Linear(28, 300),
        #     nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)
        # p.to(device)
        # print(' ')

def initialize(data, methods, prob, x0Type, HProp_all, batchsize, 
               algPara, learningRate, 
               plotAll=False, lamda=1, seed=None): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    regType: type of regularization 
    x0Type: type of starting point
    HProp_all: all Hessian proportion for sub-sampling methods
    batchsize: batchsize of first order algorithms
    algPara: a class that contains:
        mainLoopMaxItrs: maximum iterations for main loop
        funcEvalMax: maximum oracle calls (function evaluations) for algorithms
        innerSolverMaxItrs: maximum iterations for inner (e.g., CG) solvers
        lineSearchMaxItrs: maximum iterations for line search methods
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        innerSolverTol: inexactness tolerance of inner solvers
        beta: parameter of Armijo line-search
        beta2: parameter of Wolfe curvature condition (line-search)
        show: print result for every iteration
    learningRate: learning rate of the first order algorithms
    lamda: parameter of regularizer
    plotAll: plot 3 iterations plots as well
    """
    print('Initialization...')
    prob = prob[0]
    x0Type = x0Type[0]
        
    print('Problem:', prob, end='  ')
    if hasattr(algPara, 'cutData'):
        print('Data-size using: ', algPara.cutData)
    print('regulization = %8s' % algPara.regType, end='  ')
    print('innerSolverMaxItrs = %8s' % algPara.innerSolverMaxItrs, end='  ')
    print('lineSearchMaxItrs = %8s' % algPara.lineSearchMaxItrs, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('innerSolverTol= %8s' % algPara.innerSolverTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
    algPara.regType = algPara.regType[0]
    #smooth regulizer      
    if algPara.regType == 'None':
        reg = None
    if algPara.regType == 'Convex':
        reg = lambda x: regConvex(x, lamda)
    if algPara.regType == 'Nonconvex':
        reg = lambda x: regNonconvex(x, lamda)
    if algPara.regType == 'Nonsmooth':
        reg = None
          
    filename = '%s_%s_reg_%s_Itr_%s_x0_%s_reg_%s_ZOOM_%s_Nu_%s_seed_%s' % (
            prob, data, algPara.regType, algPara.mainLoopMaxItrs, x0Type, lamda, 
            algPara.zoom, algPara.student_t_nu, algPara.seed) 
        
#    filename = '%s_reg_%s_%s_FE_%s_solItr_%s_x0_%s_subH_%s_seed_%s_reg_%s_eps_%s_tol_MR_x_%s_%s_ZOOM_%s_Nonsmooth_%s' % (
#            prob, regType, data, algPara.funcEvalMax, algPara.innerSolverMaxItrs, x0Type, 
#            len(HProp_all), seed, lamda, algPara.epsilon, algPara.innerSolverTol,
#            algPara.innerSolverTol2, algPara.zoom, algPara.reg_nonsmooth)   
    mypath = filename
    print('filename', filename)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    if hasattr(algPara, 'total_run'):
        execute_pProfile(data, prob, methods, x0Type, algPara, learningRate, 
                         HProp_all, batchsize, reg, mypath, 
                         algPara.total_run)
    else:
        execute(data, methods, x0Type, algPara, learningRate, HProp_all, 
                     batchsize, reg, mypath, prob, plotAll, lamda)
    

def execute(data, methods, x0Type, algPara, learningRate, HProp_all, 
            batchsize, reg, mypath, prob, plotAll, lamda):  
    """
    Excute all methods/problems with 1 total run and give plots.
    """            
    
    if prob == 'cutest':
        p = pycutest.import_problem(data[0])
        l = p.n     
        x0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType)  
        print('Samples, Dimensions', p.m, p.n)
        
        obj = lambda x, control=None: get_obj_cutest(x, p, control)
        obj_mini_g = None
        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, learningRate, HProp_all, 
                batchsize, obj_mini_g, mypath)
    
        showFigure(methods_all, record_all, prob, mypath, plotAll)
    else:
        data_dir = '../Data'  
#        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        print('device', device)
        if prob == 'Auto-encoder':
            # tensor = tensor.to(device)
            data_dir = '../Data'
            data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            
            if data[0] == 'cifar10':
                train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                                 transform=data_tf, download=True)   
                model = auto_Encoder_CIFAR10() 
                if not hasattr(algPara, 'cutData'):
                    algPara.cutData = 50000
                
            if data[0] == 'mnist':
                train_dataset = datasets.MNIST(root=data_dir, train=True, 
                                                 transform=data_tf, download=True)   
                model = auto_Encoder_MNIST() 
                if not hasattr(algPara, 'cutData'):
                    algPara.cutData = 60000
            # train_dataset = datasets.MNIST(root=data_dir, train=True, 
            #                                transform=data_tf, download=True)
            train_loader = DataLoader(train_dataset, shuffle=False, 
                                      batch_size=algPara.cutData, drop_last=True)
            
            # def AE_Call(train_Set, model, device):
            model.to(device)
            # model_para = init_all(model, torch.nn.init.constant_, 0.)
            
            if x0Type == 'randn':
                init_all(model, torch.nn.init.normal_, mean=0., std=1.) 
            if x0Type == 'rand':
                init_all(model, torch.nn.init.uniform_, 0, 1.)
            if x0Type == 'ones':
                init_all(model, torch.nn.init.constant_, 1.)
            if x0Type == 'zeros':
                init_all(model, torch.nn.init.constant_, 0.)
            
            criterion = nn.MSELoss()
            # criterion = nn.MSELoss(reduction='elementwise_mean')
            # optimizier = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
            # for i in range(5):
            for img, _ in train_loader:
                # print(' 1')
                img = img.view(img.size(0), -1)
                # img = img[:, 200:500]
                img = img.to(device)
                # img = img.requires_grad_()
                # @profile
                def obj_AE(img, x, model, arg=None, reg=None):
                    # img = img.cuda().requires_grad_()
                    # print(output.norm())
                    # x0 = parameters_to_vector(model.parameters())
                    # print(parameters_to_vector(model.parameters()).norm())
                
                    vector_to_parameters(x, model.parameters())
                    output = model(img)
                    if reg == None:
                        reg_f = 0
                        reg_g = 0
                        reg_Hv = lambda v: 0
                    else:
                        w = parameters_to_vector(model.parameters())
                        reg_f, reg_g, reg_Hv = reg(w)
                    # print(para)
                    # print(model.decoder[2].bias)
                    model.zero_grad()                    
                    # img.requires_grad = False
                    if arg == 'f':
                        with torch.no_grad():
                            f = criterion(output, img)
                            f = f.detach()
                            # output = 0
                        # del img.grad
                        return f + reg_f
                    else:          
                        f = criterion(output, img)
                        f.backward(create_graph=True)
                        
                    if arg == 'fg':
                        g = torch.autograd.grad(
                            f, model.parameters(), create_graph=False, retain_graph=False)
                        f = f.detach()
                        g = parameters_to_vector(g).detach()
                        # del img.grad
                        # output = 0
                        return f + reg_f, g + reg_g
                    else:
                        g = torch.autograd.grad(
                            f, model.parameters(), create_graph=True, retain_graph=True)
                        g = parameters_to_vector(g)
                        
                    Hv = lambda v: parameters_to_vector(torch.autograd.grad(
                        g, model.parameters(), grad_outputs=v, 
                        create_graph=False, retain_graph=True)).detach() + reg_Hv(v)
                    
                    f = f.detach()
                    return f + reg_f, g.detach() + reg_g, Hv
                obj = lambda x, control=None: obj_AE(img, x, model, 
                                                     control, reg)
                break
            x0 = parameters_to_vector(model.parameters()).detach()
            x0 = x0/algPara.zoom
            algPara.L_g=1E6
            obj_mini_g = None
            methods_all, record_all = run_algorithms(
                    obj, x0, methods, algPara, learningRate, HProp_all, 
                    batchsize, obj_mini_g, mypath)
    
            showFigure(methods_all, record_all, prob, mypath, plotAll)
            # f, grad, Hvp = AE_Call(train_loader, model, device)
        else:
            if prob == 'toy_fun':
                from toy_fun import log_fun_old
                l = algPara.L
                X = torch.randn(l,l, dtype=algPara.dType)
            else:
                print('Dataset:', data[0])
                # from libsvm.svmutil import svm_read_problem                
                # if data[0] == 'real-sim':
                #     Y, X = svm_read_problem(r'../Data/real-sim/real-sim')
                #     print(' ')
                if data[0] == 'cifar10':
                    train_Set = datasets.CIFAR10(data_dir, train=True,
                                                transform=transforms.ToTensor(), 
                                                download=True)  
                    (n, r, c, rgb) = train_Set.data.shape
                    d = rgb*r*c
                    X = train_Set.data.reshape(n, d)
                    X = X/255
                    total_C = 10
                    X = torch.DoubleTensor(X)
                    Y_index = train_Set.targets  
                    Y_index = torch.DoubleTensor(Y_index)
                if data[0] == 'mnist':
                    train_Set = datasets.MNIST(data_dir, train=True,
                                                transform=transforms.ToTensor(), 
                                                download=True)
                    (n, r, c) = train_Set.data.shape
                    d = r*c
                    X = train_Set.data.reshape(n, d)
                    X = X/255
                    total_C = 10
                    X = X.double()
                    Y_index = train_Set.targets  
                    Y_index = Y_index.double()
                
                if data[0] == 'gisette' or data[0] == 'arcene':
                    train_X, train_Y, test_X, test_Y, idx = loaddata(data_dir, data[0])
                    X = torch.from_numpy(train_X).double()
                    X = X/torch.max(X)
                    Y = torch.from_numpy(train_Y).double()
                    Y_index = Y+1
                    d = X.shape[1]
                
                if prob == 'nls' or 'logitreg' or 'student_t':
                    # Y = (Y_index==1)*1
#                    Y = (Y_index<2).float()*1 + (Y_index==2).float()*1 + (Y_index>7).float()*1
                    Y = (Y_index%2==0).double()*1
#                    Y = (Y_index>=5).double()*1
#                    Y = (Y_index<5).double()*1
                    l = d
                if prob == 'softmax':
                    I = torch.eye(total_C, total_C - 1)
                    Y = I[np.array(Y_index), :]
                    l = d*(total_C - 1)
                    
#                if not hasattr(algPara, 'cutData'):
                spnorm = np.linalg.norm(X, 2)
                if prob == 'softmax' or prob == 'logitreg':
                    algPara.L_g=spnorm**2/6/X.shape[0] + lamda
                if prob == 'nls':
                    algPara.L_g=spnorm**2/4/X.shape[0] + lamda
                if prob == 'student_t':
                    algPara.L_g=2*spnorm**2/X.shape[0]/algPara.student_t_nu + lamda
                    
                X = X.to(device)
                Y = Y.to(device)
                
                for i in range(algPara.nperseed):
                    # print(seed)
                    curr_seed = algPara.seed + i
                    print(curr_seed)
                    torch.manual_seed(curr_seed)
                    np.random.seed(curr_seed)   
                    filename = mypath + '_seed_%s' % (curr_seed) 
                    
            #    filename = '%s_reg_%s_%s_FE_%s_solItr_%s_x0_%s_subH_%s_seed_%s_reg_%s_eps_%s_tol_MR_x_%s_%s_ZOOM_%s_Nonsmooth_%s' % (
            #            prob, regType, data, algPara.funcEvalMax, algPara.innerSolverMaxItrs, x0Type, 
            #            len(HProp_all), seed, lamda, algPara.epsilon, algPara.innerSolverTol,
            #            algPara.innerSolverTol2, algPara.zoom, algPara.reg_nonsmooth)   
                    mypath2 = mypath + '/' + filename
                    print('filename', filename)
                    if not os.path.isdir(mypath2):
                       os.makedirs(mypath2)
                    
#                    if hasattr(algPara, 'cutData') and X.shape[0] >= algPara.cutData:
#                        index = np.random.choice(np.size(X,0), algPara.cutData, replace = False)
#                        X = X[index,:]
#                        Y = Y[index]  
                        
#                    spnorm = np.linalg.norm(X, 2)
#                    if prob == 'softmax' or prob == 'logitreg':
#                        algPara.L_g=spnorm**2/6/X.shape[0] + lamda
#                    if prob == 'nls':
#                        algPara.L_g=spnorm**2/4/X.shape[0] + lamda
#                    if prob == 'student_t':
#                        algPara.L_g=2*spnorm**2/X.shape[0]/algPara.student_t_nu + lamda
                    print('Lipschiz', algPara.L_g)
    #                algPara.gradTol = min(algPara.gradTol, torch.sqrt(2*algPara.L_g*1E-15))
    #                algPara.gradTol = np.sqrt(2*algPara.L_g*1E-16)
                    print('Original_Dataset_shape:', X.shape, end='  ') 
                    # Y_index = train_Set.targets
                    # I = torch.eye(total_C, total_C - 1)
                    # train_Y = I[Y_index, :]
                    # train_X = torch.from_numpy(train_X)
                    # train_Y = torch.from_numpy(train_Y)       
                    # train_Y.to(device)         
        #    train_X = scale_train_X(train_X, standarlize=False, normalize=False) 
            
                    index_batch = np.random.choice(np.size(X,0), int(
                            batchsize*(X.shape[0])), replace = False)
                            
                    if prob == 'softmax':      
                        # X, Y, l = sofmax_init(train_X, train_Y)    
                        obj = lambda x, control=None, HProp=1: softmax(
                                X, Y, x, HProp, control, reg)  
                        mini_batch_X = X[index_batch,:]      
                        mini_batch_Y = Y[index_batch]
                        obj_mini_g = lambda x: softmax(
                                mini_batch_X, mini_batch_Y, x, arg='g', reg=reg)
                        
                    if prob == 'nls':
                        # X, Y, l = nls_init(train_X, train_Y, idx=5)
                        obj = lambda x, control=None, HProp=1: least_square(
                                X, Y, x, HProp, control, reg)
                        mini_batch_X = X[index_batch,:] 
                        mini_batch_Y = Y[index_batch]
                        obj_mini_g = lambda x: least_square(mini_batch_X, mini_batch_Y, x, 
                                                            arg='g', reg=reg)
                        
                    if prob == 'student_t':
                        # X, Y, l = nls_init(train_X, train_Y, idx=5)
                        obj = lambda x, control=None, HProp=1: student_t(
                                X, Y, x, nu=algPara.student_t_nu, HProp=HProp, 
                                arg=control, reg=reg)
                        mini_batch_X = X[index_batch,:] 
                        mini_batch_Y = Y[index_batch]
                        obj_mini_g = lambda x: student_t(mini_batch_X, mini_batch_Y, x,
                                                         nu=algPara.student_t_nu, 
                                                         arg='g', reg=reg)
                        
                    if prob == 'logitreg':
                        # X, Y, l = nls_init(train_X, train_Y, idx=5)
                        obj = lambda x, control=None, HProp=1: logitreg(
                                X, Y, x, HProp, control, reg)
                        mini_batch_X = X[index_batch,:] 
                        mini_batch_Y = Y[index_batch]
                        obj_mini_g = lambda x: logitreg(mini_batch_X, mini_batch_Y, x, 
                                                            arg='g', reg=reg)
                        
                    x0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType, dvc = device)   
                        
                    methods_all, record_all = run_algorithms(
                            obj, x0, methods, algPara, learningRate, HProp_all, 
                            batchsize, obj_mini_g, mypath2)
                    if methods_all[0] != 'abort': # for acc use
                        showFigure(methods_all, record_all, prob, mypath2, plotAll)
                      
#    print(x0.T.dot(x0))
        
#    methods_all, record_all = run_algorithms(
#            obj, x0, methods, algPara, learningRate, HProp_all, 
#            batchsize, obj_mini_g, mypath)
#    
#    showFigure(methods_all, record_all, prob, mypath, plotAll)

def get_obj_cutest(x, p, arg):
    dType = p.dtype
    if arg is None:
        out = [torch.tensor(p.obj(x)).to(dType), 
               torch.tensor(p.obj(x, gradient=True)[1]).to(dType), 
               lambda v: torch.tensor(p.hprod(v, x=x)).to(dType)]
    if arg == 'fg':
        out = [torch.tensor(p.obj(x)).to(dType), 
               torch.tensor(p.obj(x, gradient=True)[1]).to(dType)]
    if arg == 'f':
        out = torch.tensor(p.obj(x)).to(dType)
    if arg == 'g':
        out = torch.tensor(p.obj(x, gradient=True)[1]).to(dType)
    # if arg is None:
    #     out = [p.obj(x[:,0], gradient=True)[0], p.obj(x[:,0], gradient=True)[1].reshape(
    #                 len(x[:,0]),1), lambda v: p.hprod(v[:,0], x=x[:,0]).reshape(len(v[:,0]),1)]
    # if arg == 'fg':
    #     out = [p.obj(x[:,0], gradient=True)[0], p.obj(x[:,0], gradient=True)[1].reshape(len(x[:,0]),1)]
    # if arg == 'f':
    #     out = p.obj(x[:,0], gradient=True)[0]
    # if arg == 'g':
    #     out = p.obj(x[:,0], gradient=True)[1].reshape(len(x[:,0]),1)
    return out
  
def execute_pProfile(data, prob, methods, x0Type, algPara, learningRate, HProp_all, 
                     batchsize, reg, mypath, total_run): 
    """
    Excute all methods/problems with multiple runs. Compare the best results
    and give performance profile plots between methods.
    
    Record a pProfile matrix s.t., every row are the best (f/g/error) result 
        of all listed methods. e.g., 500 total_run of 5 different 
        optimisation methods will be a 500 x 5 matrix.
    """           
    
    mypath_pProfile = 'pProfile'
    if not os.path.isdir(mypath_pProfile):
       os.makedirs(mypath_pProfile) 
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    assert prob == 'cutest'
    print('device', device)
    if data[0] == 'UR':
        data = pycutest.find_problems(constraints='U', origin='R', userN=False)  
#            data = pycutest.find_problems(constraints='U', userN=True)  
        data = ['COOLHANSLS', 'MEYER3', 'PALMER1C', 'PALMER1D', 'PALMER2C', 
                'PALMER3C', 'PALMER4C', 'PALMER5C', 'PALMER5D', 'PALMER6C', 
                'PALMER7C', 'PALMER8C',] #12
    if data[0] == 'U':
#            data1 = pycutest.find_problems(objective='Q', constraints='U', 
#                                           regular=True, n=[100, 1e10])  
        
#            data1 = pycutest.find_problems(objective='Q', constraints='U', 
#                                           regular=True)  
#            print(data1)
#            data2 = pycutest.find_problems(objective='S', constraints='U', 
#                                           regular=True, n=[100, 1e10])  
#            print(data2)
#            data3 = pycutest.find_problems(objective='O', constraints='U', 
#                                           regular=True, n=[100, 1e10])  
#            print(data3)
        data = [
                # d >= 100 ## 121
                #     'TRIDIA', 'TESTQUAD', 'DQDRTIC', 'DIXON3DQ', 'PENALTY2', 'KSSLS', 'SSBRYBND',
                #     'MODBEALE', 'ARGTRIGLS', 'PENALTY1',
                #     'LUKSAN22LS', 'LUKSAN16LS', 'EIGENALS', 'MSQRTBLS', 'INTEQNELS', 
                #     'MNISTS5LS', 'BROYDNBDLS', 'GENROSE', 'NONDIA', 'ARGLINB', 
                #     'SROSENBR', 'EIGENCLS', 'COATING', 'YATP2LS', 'LUKSAN21LS', ##TR1 ##1
                #     'YATP2CLS', 'BROWNAL', ##2
                #   'MANCINO', 'SPIN2LS', 'LUKSAN15LS', 'CHAINWOO', 'ARGLINC', 'CYCLOOCFLS',
                # 'MNISTS0LS', 'OSCIPATH', 'BDQRTIC', 'BRYBND', 'LUKSAN11LS', 'EIGENBLS', 'MSQRTALS', ##3
                # 'SPMSRTLS', 'QING', 'EXTROSNB', 'WOODS', 'TQUARTIC', 'FREUROTH',
                # 'ARGLINA', 'BROYDN3DLS', ##4 
                # 'LUKSAN17LS', 'LIARWHD', 'NONMSQRT',
                # 'OSCIGRAD', 'FMINSURF', 'SENSORS', 'DIXMAANB', 'SINQUAD', 'POWER', 'DIXMAANH',
                # 'FLETCHBV', 'QUARTC', 'TOINTGSS', 'EG2', 'CURLY10', 'DIXMAANJ', 'DIXMAANE', 'FMINSRF2', ##5
                # 'PENALTY3', 'SCOSINE', 'ARWHEAD', ##6
                # 'NCB20', 'NCB20B', 'FLETCBV2', 'DIXMAANC', 'SSCOSINE', 'SCHMVETT', 'VARDIM', 'BOXPOWER',
                # 'POWELLSG', 'DIXMAANM', 'NONDQUAR', ##7
                # 'FLETBV3M', 'COSINE', 'DIXMAANP',
                # 'GENHUMPS', 'DIXMAANF', 
                # 'DIXMAANN', 'CURLY20', 'DIXMAANI',
                # 'DIXMAANA', 'FLETCBV3', 'NONCVXUN', 'NONCVXU2', 'DIXMAANG', 'ENGVAL1',
                # 'EDENSCH', 'DIXMAANO', 'FLETCHCR', 'DIXMAANK', ##8
                # 'SPARSINE', 'DQRTIC', 'DIXMAANL', 'BROYDN7D', 'INDEFM', 
                # 'BOX', 'VAREIGVL', 'CRAGGLVY', 'CURLY30', 'DIXMAAND', ##9 ##TR2
                
                #rand+
                # 'YATP1LS',  #randn?
                # 'YATP1CLS', #? daozhele
                # 'INDEF', #?
                # 'SPARSQUR', #? TR?
                
                #??gradnorm
                # 'SPINLS', #?
                # 'SCURLY30', #?
                # 'SCURLY20', #?
                # 'SCURLY10', # ?
                
                #rand ??
                # 'JIMACK', #? RAND?
                # 'BA-L21LS', #? da
                # 'BA-L49LS', #? #TR? edit TR include ? + TR?
                
                
                # 'BA-L52LS', # ? and da TR not finishd only this one left
#                    ## 10 <= d < 100 ## 32 + 121 = 153
                # 'HILBERTB', 
                # 'TOINTQOR', #'DIAMON2DLS', 'DIAMON3DLS', 'OSBORNEB', 'LUKSAN14LS', ##1
                # 'DMN15103LS', 'DMN15333LS', 'ERRINROS', 'BA-L1SPLS', 
                # 'LUKSAN12LS', 'BA-L1LS', 'HYDC20LS', 'HATFLDGLS', 'HYDCAR6LS', 
                # 'ERRINRSM', 'DMN37142LS', 'TRIGON2', 'LUKSAN13LS', 'WATSON', 
                # 'DMN37143LS', 'CHNRSNBM', 'VANDANMSLS', 'STRTCHDV', ##2
                # 'CHNROSNB', 'TOINTPSP', 'TOINTGOR', ##3
                
                #rand+
                # 'DMN15332LS', #?
                
                #rand ?
                # 'METHANL8LS', #?
                # 'METHANB8LS', #?
                # 'STRATEC', # - inf unbound below
                # 'PARKCH',  # - inf unbound below
                ## 0 < d < 10 ## 127 + 153 = 280
                # 'PALMER5C', 'ZANGWIL2', 'HILBERTA', 'PALMER1C', #CG-pert insane, why
                # 'PALMER4C', 'PALMER5D', 
                # 'PALMER7C', 'PALMER1D', 
                # 'PALMER2C', 'PALMER8C', 'PALMER3C', 'DENSCHND', 
                # 'DENSCHNC', 'RAT42LS', ##1
                # 'POWELLSQLS', 'PRICE4', 
                # 'LANCZOS1LS', 
                # 'DENSCHNF', 'EXP2', 'VESUVIALS', 'BROWNDEN', ##2
                # 'ROSENBR', 'ENSOLS', 'HEART8LS', 
                'RAT43LS', #1E6?
                'THURBERLS',#1E6?
                # 'BARD', 'MGH10SLS', 'S308NE', ##3
                # 'EXPFIT', 
                # 'GULF', 'EGGCRATE', 'BOXBODLS', 'S308', 
                # 'CHWIRUT2LS', #1E5?
                'PALMER6C', 'POWERSUM', 'HEART6LS', 'HATFLDE', 
                'HAHN1LS', 'FBRAIN3LS', 'HIMMELBF', 'BROWNBS', 'LSC1LS', 'SINEVAL', 'LANCZOS3LS', 
                'VIBRBEAM', 'MGH09LS', 'JENSMP', 'MGH10LS', 'GROWTHLS', 'MISRA1BLS', 'BEALE', ##4
                'DANIWOODLS', 'PRICE3', 'POWELLBSLS', 'LANCZOS2LS', 'VESUVIOLS', 'HATFLDFLS', 'MGH17LS', 
                'GBRAINLS', 'COOLHANSLS', 'SSI', 'GAUSS2LS', 'MEYER3', 'KIRBY2LS', 'YFITU', 
                  'MISRA1DLS', 'ENGVAL2', 'CLUSTERLS', 'WAYSEA1', 'MISRA1ALS', 'RECIPELS', 
                'HELIX', 'ROSZMAN1LS', 'LSC2LS', 'MUONSINELS', 'ELATVIDU', 'NELSONLS', 'GAUSS3LS', 
                'CLIFF', 'DJTL', 'BRKMCC', 'LOGHAIRY', 'ROSENBRTU', 'HIMMELBH', 'MARATOSB', 'DENSCHNA',
                'HAIRY', 'ALLINITU', 'SISSER', 'HIMMELBB', 'HUMPS', ##5
                'MEXHAT', 'HIMMELBCLS', 'HIMMELBG', 'SNAIL', 
                'GAUSS1LS', 'JUDGE', 'CERI651DLS', 'GAUSSIAN', 'DEVGLA2NE', 'KOWOSB', 'BOX3', 
                'DENSCHNB', 'CHWIRUT1LS', 'WAYSEA2', 'DENSCHNE', 'HATFLDD', 'CUBE', 'CERI651BLS', 
                'VESUVIOULS', 'ECKERLE4LS', 'CERI651ELS', 'BIGGS6', 'HATFLDFL', 'STREG', ##7
                
                ###rand+
                # 'CERI651ALS', #? rand+
                # 'BENNETT5LS',#? rand+
                # 'DEVGLA2', #?   rand+ 
                # 'CERI651CLS', #? rand+
                
                ##rand/1e5+Wolfe amax=100
                # 'OSBORNEA', #rand ?
                # 'DEVGLA1', #?
                # 'DANWOODLS', #?
                # 'MISRA1CLS', #?
                # 'AKIVA'#TR? ##6
                
                
                # 'HIELOW', #bug2 Unbounded below, NC detected f=-inf. No worries for Sol.
                ]
                
    if data[0] == 'UO':
        data = pycutest.find_problems(
                constraints='U', regular=True, objective='O', userN=False)
#            data = ['AKIVA', 'ALLINITU', 'BRKMCC', 'CLIFF', 'DENSCHNA', 'DJTL', 
#                    'HAIRY', 'HIELOW', 'HIMMELBB', 'HIMMELBCLS', 'HIMMELBG', 
#                    'HIMMELBH', 'HUMPS', 'JIMACK', 'LOGHAIRY', 'MARATOSB', 
#                    'MEXHAT', 'PARKCH', 'ROSENBRTU', 'SISSER', 'SNAIL', 
#                    'STRATEC', 'TOINTGOR', 'TOINTPSP'] #24
        data = ['ALLINITU', 'BRKMCC', 'CLIFF', 'DENSCHNA', 'DJTL', 
                'HAIRY', 'HIMMELBB', 'HIMMELBCLS', 'HIMMELBG', 
                'HIMMELBH', 'HUMPS', 'LOGHAIRY', 'MARATOSB', 
                'MEXHAT', 'ROSENBRTU', 'SISSER', 'SNAIL', 
                'TOINTGOR', 'TOINTPSP'] #20, 'HIELOW'6
#            data = ['CLIFF', 'TOINTPSP']
    for i in range(len(data)):
        print('Current Problem', data[i], i)
        p = pycutest.import_problem(data[i])
        l = p.n     
        x0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType, dvc = device)  
        print('Samples, Dimensions', p.m, p.n)
        obj = lambda x, control=None: get_obj_cutest(x, p, control)
            
    #############################################################################
        obj_mini_g = None

        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, learningRate, HProp_all, 
                batchsize, obj_mini_g, mypath)            
        
        number_of_methods = len(methods_all)     
        pProfile_fi = np.zeros((1,number_of_methods))
        pProfile_gi = np.zeros((1,number_of_methods)) 
        
        for m in range(number_of_methods):
            record_matrices_i = record_all[m]
            pProfile_fi[0,m] = record_matrices_i[-1,0]
            pProfile_gi[0,m] = record_matrices_i[-1,1]
            #JIMACK 56
        if i == 0:     
            pProfile_f = pProfile_fi
            pProfile_g = pProfile_gi
            with open(os.path.join(mypath_pProfile, 'methods.txt'), \
                      'w') as myfile:
                for method in methods_all:
                    myfile.write('%s\n' % method)
        else:
            pProfile_f = np.append(pProfile_f, pProfile_fi, axis=0)
            pProfile_g = np.append(pProfile_g, pProfile_gi, axis=0)
            
        np.savetxt(os.path.join(mypath_pProfile, 'problem.txt'), \
                   data[:i+1], delimiter="\n", fmt="%s")
        np.savetxt(os.path.join(mypath_pProfile, 'objVal.txt'), \
                   pProfile_f, delimiter=',')
        np.savetxt(os.path.join(mypath_pProfile, 'gradNorm.txt'), \
                   pProfile_g, delimiter=',')
       
    
    figsz = (6,4)
    mydpi = 200      
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_f, ylabel='F')
    fig1.savefig(os.path.join(mypath_pProfile, 'objVal'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_g, ylabel='GradientNorm')
    fig2.savefig(os.path.join(mypath_pProfile, 'gradNorm'), dpi=mydpi)
    
    with open(os.path.join(mypath_pProfile, 'methods.txt'), 'w') as myfile:
        for method in methods_all:
            myfile.write('%s\n' % method)
    np.savetxt(os.path.join(mypath_pProfile, 'objVal.txt'), \
               pProfile_f, delimiter=',')
    np.savetxt(os.path.join(mypath_pProfile, 'gradNorm.txt'), \
               pProfile_g, delimiter=',')
    
        
        
def run_algorithms(obj, x0, methods, algPara, learningRate, HProp_all, 
                   batchsize, obj_mini_g, mypath):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []            
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join(mypath, filename+'.txt'), myrecord.cpu(), delimiter=',')    
    if 'Newton_MR_invex' in methods:  
        print(' ')
        myMethod = 'Newton_MR_invex'
        for i in range(len(HProp_all)):
            HProp = HProp_all[i]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp_all[i]*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_MR_invex(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
        
    if 'Newton_CG' in methods:
        print(' ')
        myMethod = 'Newton_CG'
        for i in range(len(HProp_all)):
            HProp = HProp_all[i]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp_all[i]*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
            
    if 'Gauss_Newton' in methods:
        print(' ')
        myMethod = 'Gauss_Newton'
        print(myMethod)
        record_all.append(myMethod)
        x, record = Gauss_Newton(
                obj, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                algPara.show)
        np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Momentum' in methods:
        print(' ')
        myMethod = 'Momentum'
        print(myMethod)
        record_all.append(myMethod)
        x, record = MomentumSGD(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Momentum, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'Momentum.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adagrad' in methods:
        print(' ')
        myMethod = 'Adagrad'
        print(myMethod)
        record_all.append(myMethod)
        x, record = Adagrad(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adagrad, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'Adagrad.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adadelta' in methods:
        print(' ')
        myMethod = 'Adadelta'
        print(myMethod)
        record_all.append(myMethod)
        x, record = Adadelta(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adadelta, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'Adadelta.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'RMSprop' in methods:
        print(' ')
        myMethod = 'RMSprop'
        print(myMethod)
        record_all.append(myMethod)
        x, record = RMSprop(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.RMSprop, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'RMSprop.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adam' in methods:
        print(' ')
        myMethod = 'Adam'
        print(myMethod)
        record_all.append(myMethod)
        x, record = Adam(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adam, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'Adam.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'SGD' in methods:
        print(' ')
        myMethod = 'SGD'
        print(myMethod)
        record_all.append(myMethod)
        x, record = SGD(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.SGD, algPara.gradTol, 
                algPara.show)
        np.savetxt(os.path.join(mypath, 'SGD.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Newton_MR' in methods:  
        print(' ')
        myMethod = 'Newton_MR'
        for i in range(len(HProp_all)):
            HProp = HProp_all[i]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp_all[i]*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_MR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                    algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                    algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta0, 
                    algPara.epsilon, algPara.show, record_txt)
            record_txt(myMethod, record)
#            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
            
                
    if 'L_BFGS' in methods:
        print(' ')
        myMethod = 'L_BFGS'
        print(myMethod)
        record_all.append(myMethod)
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.L, algPara.beta, 
                algPara.beta2, algPara.show, record_txt)
        np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record, delimiter=',')
        record_all.append(record)
            
    if 'AndersonAcc' in methods:
        print(' ')
        myMethod = 'AndersonAcc'
#        arg = ['GD', 'pure', 'residual', 'general']
#        arg = ['GD', 'pure', 'residual']
#        arg = ['pure', 'residual', 'general']
#        arg = ['GD', 'residual']
#        arg = ['general', 'pure', 'residual', 'GD']
#        arg = ['general', 'pure']
#        arg = ['pure', 'general']
        arg = ['pure']
#        arg = ['general']
        flag = 1
        i = 0
        maxIt = algPara.mainLoopMaxItrs
        while flag == 1 and i < (len(arg)):
            if i > 0:
                maxIt = record.shape[0]
            arg_i = arg[i]            
            myMethod = 'AndersonAcc_%s' % (arg_i)
            print(myMethod)
            x, record = AndersonAcc(
                    obj, x0, algPara.Andersonm, algPara.L_g, maxIt, 
                    algPara.funcEvalMax, algPara.gradTol, algPara.show, arg_i, record_txt)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record.cpu(), delimiter=',')
            i += 1
#            if arg_i == 'general':
#                if sum(record[:,-1][-algPara.Andersonm:]) != 0:
#                    flag = 0
#                    shutil.rmtree(mypath)
#                    myMethod = 'abort'
            record_all.append(myMethod)
            record_all.append(record.cpu())
    
    if 'Newton_CR' in methods:  
        print(' ')
        myMethod = 'Newton_CR'        
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.beta2, algPara.epsilon, 
                    algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
    
    if 'Damped_Newton_CG' in methods:  
        print(' ')
        myMethod = 'Newton_CG_Pert'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Damped_Newton_CG(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol2, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.epsilon, 
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
            
    if 'Newton_MR_TR_NS' in methods:  
        print(' ')         
        arg = ['TR', 'TR_NS']
#         arg = ['TR']
#        arg = ['TR_NS']
        for i in range(len(arg)):
            arg_i = arg[i]
            myMethod = 'Newton_MR_%s' % (arg_i)
            for j in range(len(HProp_all)):
                HProp = HProp_all[j]
                if HProp != 1:
                    obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                    myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
                else:
                    obj_subsampled = obj
                print(myMethod)
                record_all.append(myMethod)
                x, record = Newton_MR_TR(
                        obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                        algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                        algPara.lineSearchMaxItrs, algPara.gradTol,
                        algPara.Delta, algPara.Delta_max, algPara.gama1, algPara.gama2, algPara.eta, 
                        algPara.show, arg_i, record_txt)
                record_txt(myMethod, record)
                record_all.append(record)
    
    if 'Newton_CG_TR' in methods:  
        print(' ')
        myMethod = 'Newton_CG_TR'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_TR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol1, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
    
    if 'Newton_CG_TR_Pert' in methods:  
        print(' ')
        myMethod = 'Newton_CG_TR_Pert'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_TR_Pert(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol2, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.epsilon, 
                    algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            record_txt(myMethod, record)
            record_all.append(record)
            
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    
    return methods_all, record_all

        
def sofmax_init(train_X, train_Y):
    """
    Initialize data matrix for softmax problems.
    For multi classes classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Classes = sorted(set(train_Y))
    Total_C  = len(Classes)
    if Total_C == 2:
        train_Y = (train_Y == 1)*1
    l = d*(Total_C-1)
    I = np.ones(n)
    
    X_label = np.array([i for i in range(n)])
    Y = sparse.coo_matrix((I,(X_label, train_Y)), shape=(
            n, Total_C)).tocsr().toarray()
    Y = Y[:,:-1]
    return train_X, Y, l    

        
def nls_init(train_X, train_Y, idx=5):
    """
    Initialize data matrix for non-linear least square problems.
    For binary classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
        idx: a number s.t., relabelling index >= idx classes into 1, the rest 0. 
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Y = (train_Y >= idx)*1 #bool to int
    Y = Y.reshape(n,1)
    l = d
    return train_X, Y, l

def scale_train_X(train_X, standarlize=False, normalize=False): 
    """
    Standarlization/Normalization of trainning DATA.
    """
    if standarlize:
        train_X = preprocessing.scale(train_X)            
    if normalize:
        train_X = preprocessing.normalize(train_X, norm='l2')
    return train_X

    
def generate_x0(x0Type, l, zoom=1, dType=torch.float64, dvc = 'cpu'):    
    """
    Generate different type starting point.
    """
    if x0Type == 'randn':
        x0 = torch.randn(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'rand':
        x0 = torch.rand(l, dtype=dType, device=dvc)/zoom
    if x0Type == 'ones':
        x0 = torch.ones(l, dtype=dType, device=dvc)
    if x0Type == 'zeros':
        x0 = torch.zeros(l, dtype=dType, device=dvc)
    return x0