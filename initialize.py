import numpy as np
import torch
from torch import nn
from optim_algo import (Naive_Newton_MR, L_BFGS, Newton_MR, Newton_CR, Newton_CG_LS, 
                        Newton_CG_TR_Steihaug, Newton_CG_TR_Pert, Nonlinear_CG,
                        )
from loaddata import loaddata
from showFigure import showFigure
from sklearn import preprocessing
from regularizer import regConvex, regNonconvex
from student_t import student_t
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
from pProfile import pProfile
from softmax import softmax
from logitreg import logitreg
from least_square import least_square
from scipy import sparse
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

        
class auto_Encoder_CIFAR10(nn.Module):
    def __init__(self):
        super(auto_Encoder_CIFAR10, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(3*32*32, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 16),
                                      nn.Tanh(),
                                      nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16),
                                      nn.Tanh(),
                                      nn.Linear(16, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 3*32*32),
                                      )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class auto_Encoder_MNIST(nn.Module):
    def __init__(self):
        super(auto_Encoder_MNIST, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 16),
                                      )
        self.decoder = nn.Sequential(nn.Linear(16, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 28*28),
                                      )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class auto_Encoder_STL10(nn.Module):
    def __init__(self):
        super(auto_Encoder_STL10, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(96*96*3, 150),
                                     nn.Sigmoid(),
                                     nn.Linear(150, 100),
                                     nn.Sigmoid(),
                                     nn.Linear(100, 50),
                                     nn.Sigmoid(),
                                     nn.Linear(50, 25),
                                     nn.Sigmoid(),
                                     nn.Linear(25, 6),
                                     nn.ReLU(True))
        self.decoder = nn.Sequential(nn.Linear(6, 25),
                                     nn.Sigmoid(),
                                     nn.Linear(25, 50),
                                     nn.Sigmoid(),
                                     nn.Linear(50, 100),
                                     nn.Sigmoid(),
                                     nn.Linear(100, 150),
                                     nn.Sigmoid(),
                                     nn.Linear(150, 96*96*3),
                                     nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def initialize(data, methods, prob, regType, x0Type, HProp_all, algPara, 
               plotAll=False, lamda=1): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    regType: type of regularization 
    x0Type: type of starting point
    HProp_all: all Hessian proportion for sub-sampling methods
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
    lamda: parameter of regularizer
    plotAll: plot 3 iterations plots as well
    """
    print('Initialization...')
    regType = regType[0]
    prob = prob[0]
    x0Type = x0Type[0]
        
    print('Problem:', prob, end='  ')
    if hasattr(algPara, 'cutData'):
        print('Data-size using: ', algPara.cutData)
    print('regulization = %8s' % regType, end='  ')
    print('innerSolverMaxItrs = %8s' % algPara.innerSolverMaxItrs, end='  ')
    print('lineSearchMaxItrs = %8s' % algPara.lineSearchMaxItrs, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('innerSolverTol= %8s' % algPara.innerSolverTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
        
    #smooth regulizer      
    if regType == 'None':
        reg = None
    if regType == 'Convex':
        reg = lambda x: regConvex(x, lamda)
    if regType == 'Nonconvex':
        reg = lambda x: regNonconvex(x, lamda)
          
    filename = '%s_%s_reg_%s_Orc_%s_x0_%s_reg_%s_ZOOM_%s_Nu_%s' % (
            prob, regType, data, algPara.funcEvalMax, x0Type, lamda, 
            algPara.zoom, algPara.student_t_nu) 
    mypath = filename
    print('filename', filename)
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    if hasattr(algPara, 'total_run'):
        execute_pProfile(data, prob, methods, x0Type, algPara, 
                         HProp_all, reg, mypath, 
                         algPara.total_run)
    else:
        execute(data, methods, x0Type, algPara, HProp_all, reg, 
                mypath, prob, plotAll, lamda)
    

def execute(data, methods, x0Type, algPara, HProp_all, 
            reg, mypath, prob, plotAll, lamda):  
    """
    Excute all methods/problems with 1 total run and give plots.
    """            
    
    if prob == 'cutest':
        p = pycutest.import_problem(data[0])
        l = p.n     
        x0 = generate_x0(x0Type, l, zoom=algPara.zoom, dType=algPara.dType)  
        print('Samples, Dimensions', p.m, p.n)
        
        obj = lambda x, control=None: get_obj_cutest(x, p, control)
    else:
        data_dir = '../Data'  
        if algPara.cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        print('device', device)
        if prob == 'Auto-encoder':
            # tensor = tensor.to(device)
            data_dir = '../Data'
            data_tf = transforms.Compose([transforms.ToTensor()])
            
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
        
            if data[0] == 'stl10':
                train_dataset = datasets.STL10(data_dir,split='train',
                                           transform=transforms.ToTensor(), 
                                           download=True)
                model = auto_Encoder_STL10() 
                if not hasattr(algPara, 'cutData'):
                    algPara.cutData = 5000
            train_loader = DataLoader(train_dataset, shuffle=False, 
                                      batch_size=algPara.cutData, drop_last=True)
            
            model.double()
            model.to(device)
            
            if x0Type == 'randn':
                init_all(model, torch.nn.init.normal_, mean=0., std=1.) 
            if x0Type == 'rand':
                init_all(model, torch.nn.init.uniform_, 0, 1.)
            if x0Type == 'ones':
                init_all(model, torch.nn.init.constant_, 1.)
            if x0Type == 'zeros':
                init_all(model, torch.nn.init.constant_, 0.)
            
            # criterion = nn.CrossEntropyLoss()
            # criterion.double()
            criterion = nn.MSELoss()
            for img, _ in train_loader:
                img = img.view(img.size(0), -1)
                img = img.double()
                img = img.to(device)
                def obj_AE(img, x, model, arg=None, reg=None):                
                    vector_to_parameters(x, model.parameters())
                    output = model(img)
                    if reg == None:
                        reg_f = 0
                        reg_g = 0
                        reg_Hv = lambda v: 0
                    else:
                        w = parameters_to_vector(model.parameters())
                        reg_f, reg_g, reg_Hv = reg(w)
                    model.zero_grad()  
                    if arg == 'f':
                        with torch.no_grad():
                            f = criterion(output, img)
                            f = f.detach()
                        return f + reg_f
                    else:          
                        f = criterion(output, img)
                    if arg == 'fg':
                        g = torch.autograd.grad(
                            f, model.parameters(), create_graph=False, retain_graph=False)
                        f = f.detach()
                        g = parameters_to_vector(g).detach()
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
            if algPara.normalizex0:
                x0 = x0/x0.norm()
            x0 = x0.double()
            vector_to_parameters(x0, model.parameters())
            print('dim', x0.shape, x0)
            
            methods_all, record_all = run_algorithms(
                    obj, x0, methods, algPara, HProp_all, mypath)
            
            showFigure(methods_all, record_all, prob, mypath, plotAll)
        else:
            print('Dataset:', data[0])
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
    
            if data[0] == 'stl10':
                train_Set = datasets.STL10(data_dir,split='train',
                                           transform=transforms.ToTensor(), 
                                           download=True)
                (n, r, c, rgb) = train_Set.data.shape
                d = rgb*r*c
                X = train_Set.data.reshape(n, d)
                X = X/255
                total_C = 10
                X = torch.DoubleTensor(X)
                Y_index = train_Set.labels
                Y_index = torch.DoubleTensor(Y_index)
                
            if data[0] == 'gisette' or data[0] == 'arcene':
                train_X, train_Y, test_X, test_Y, idx = loaddata(data_dir, data[0])
                # train_X = train_X[:10,:100]
                # train_Y = train_Y[:10]
                X = torch.from_numpy(train_X).double()
                X = X/torch.max(X)
                Y = torch.from_numpy(train_Y).double()
                Y_index = (Y+1)/2
                d = X.shape[1]
                
            if data[0] == '20news':
                cats = [
                        # 'alt.atheism',
                        # 'comp.graphics',
                        # 'comp.os.ms-windows.misc',
                        # 'comp.sys.ibm.pc.hardware',
                        # 'comp.sys.mac.hardware',
                        'comp.windows.x',
                        'misc.forsale',
                        'rec.autos',
                        'rec.motorcycles',
                        'rec.sport.baseball',
                        'rec.sport.hockey',
                        'sci.crypt',
                        'sci.electronics',
                        'sci.med',
                        'sci.space',
                        # 'soc.religion.christian',
                        # 'talk.politics.guns',
                        # 'talk.politics.mideast',
                        # 'talk.politics.misc',
                        # 'talk.religion.misc',
                         ]
                news = fetch_20newsgroups(subset="all", categories=cats)
                train_X, test_X, train_Y, test_Y = train_test_split(news.data, news.target)
                vectorizer = TfidfVectorizer()
                X = torch.tensor(vectorizer.fit_transform(train_X).todense())
                Y_index = torch.tensor(train_Y)
                n, d = X.shape
                
            if hasattr(algPara, 'cutData') and X.shape[0] >= algPara.cutData:
                index = np.random.choice(np.size(X,0), algPara.cutData, replace = False)
                X = X[index,:]
                Y_index = Y_index[index]    
                
            if prob == 'nls' or 'logitreg' or 'student_t':
                Y = (Y_index%2==0).double()*1
                l = d
            if prob == 'softmax':
                I = torch.eye(total_C, total_C - 1)
                Y = I[np.array(Y_index), :]
                l = d*(total_C - 1)
                
            X = X.to(device)
            Y = Y.to(device)
            print('Original_Dataset_shape:', X.shape, end='  ') 
            
            print('filename', mypath)
            if not os.path.isdir(mypath):
               os.makedirs(mypath)
            if prob == 'softmax':      
                # X, Y, l = sofmax_init(train_X, train_Y)    
                obj = lambda x, control=None, HProp=1: softmax(
                        X, Y, x, HProp=HProp, arg=control, reg=reg)  
                
            if prob == 'nls':
                # X, Y, l = nls_init(train_X, train_Y, idx=5)
                obj = lambda x, control=None, HProp=1, index=None: least_square(
                        X, Y, x, HProp=HProp, arg=control, reg=reg)
                
            if prob == 'student_t':
                # X, Y, l = nls_init(train_X, train_Y, idx=5)
                obj = lambda x, control=None, HProp=1, index=None: student_t(
                        X, Y, x, nu=algPara.student_t_nu, HProp=HProp, 
                        arg=control, reg=reg, index=index)
                
            if prob == 'logitreg':
                # X, Y, l = nls_init(train_X, train_Y, idx=5)
                obj = lambda x, control=None, HProp=1: logitreg(
                        X, Y, x, HProp=HProp, arg=control, reg=reg)
                
            x0 = generate_x0(x0Type, l, zoom=l, dType=algPara.dType) 
            x0 = x0/algPara.zoom
            if algPara.normalizex0:
                x0 = x0/x0.norm()
            x0 = x0.to(device)  
                
            methods_all, record_all = run_algorithms(
                    obj, x0, methods, algPara, HProp_all, mypath)
            
            showFigure(methods_all, record_all, prob, mypath, plotAll)

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
    return out
  
def execute_pProfile(data, prob, methods, x0Type, algPara, HProp_all, 
                    reg, mypath, total_run): 
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
            
    if algPara.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
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
#                                           regular=True)  
#            print(data1)
#            data2 = pycutest.find_problems(objective='S', constraints='U', 
#                                           regular=True, n=[100, 1e10])  
#            print(data2)
#            data3 = pycutest.find_problems(objective='O', constraints='U', 
#                                           regular=True, n=[100, 1e10])  
#            print(data3)
        data = [ # 237 problems in total, all from data1 + data2 + data3 above
                "PALMER5C",
                "ZANGWIL2",
                "HILBERTA",
                "PALMER1C",
                "PALMER4C",
                "PALMER5D",
                "PALMER7C",
                "PALMER1D",
                "PALMER2C",
                "PALMER8C",
                "PALMER3C",
                "DENSCHND",
                "DENSCHNC",
                "RAT42LS",
                "POWELLSQLS",
                "PRICE4",
                "LANCZOS1LS",
                "DENSCHNF",
                "EXP2",
                "VESUVIALS",
                "BROWNDEN",
                "ROSENBR",
                "ENSOLS",
                "HEART8LS",
                "BARD",
                "S308NE",
                "EXPFIT",
                "GULF",
                "BOXBODLS",
                "S308",
                "RAT43LS",
                "PALMER6C",
                "POWERSUM",
                "HEART6LS",
                "HATFLDE",
                "FBRAIN3LS",
                "HIMMELBF",
                "BROWNBS",
                "LSC1LS",
                "SINEVAL",
                "LANCZOS3LS",
                "MGH09LS",
                "JENSMP",
                "MGH10LS",
                "GROWTHLS",
                "MISRA1BLS",
                "BEALE",
                "DANIWOODLS",
                "PRICE3",
                "POWELLBSLS",
                "LANCZOS2LS",
                "VESUVIOLS",
                "HATFLDFLS",
                "GBRAINLS",
                "COOLHANSLS",
                "SSI",
                "GAUSS2LS",
                "MEYER3",
                "KIRBY2LS",
                "YFITU",
                "ENGVAL2",
                "CLUSTERLS",
                "WAYSEA1",
                "RECIPELS",
                "HELIX",
                "ROSZMAN1LS",
                "LSC2LS",
                "MUONSINELS",
                "ELATVIDU",
                "CLIFF",
                "BRKMCC",
                "LOGHAIRY",
                "ROSENBRTU",
                "HIMMELBH",
                "MARATOSB",
                "DENSCHNA",
                "HAIRY",
                "ALLINITU",
                "SISSER",
                "HIMMELBB",
                "MEXHAT",
                "HIMMELBCLS",
                "HIMMELBG",
                "GAUSS1LS",
                "JUDGE",
                "HILBERTB",
                "TOINTQOR",
                "DMN15333LS",
                "ERRINROS",
                "BA-L1SPLS",
                "LUKSAN12LS",
                "HATFLDGLS",
                "HYDCAR6LS",
                "ERRINRSM",
                "TRIGON2",
                "LUKSAN13LS",
                "WATSON",
                "CHNRSNBM",
                "VANDANMSLS",
                "STRTCHDV",
                "CHNROSNB",
                "TOINTPSP",
                "TOINTGOR",
                "DIAMON2DLS",
                "DIAMON3DLS",
                "LUKSAN14LS",
                "TRIDIA",
                "TESTQUAD",
                "DQDRTIC",
                "DIXON3DQ",
                "PENALTY2",
                "KSSLS",
                "MODBEALE",
                "ARGTRIGLS",
                "PENALTY1",
                "LUKSAN22LS",
                "LUKSAN16LS",
                "EIGENALS",
                "MSQRTBLS",
                "INTEQNELS",
                "MNISTS5LS",
                "BROYDNBDLS",
                "GENROSE",
                "NONDIA",
                "ARGLINB",
                "SROSENBR",
                "EIGENCLS",
                "COATING",
                "YATP2LS",
                "LUKSAN21LS",
                "YATP2CLS",
                "BROWNAL",
                "MANCINO",
                "SPIN2LS",
                "CHAINWOO",
                "ARGLINC",
                "CYCLOOCFLS",
                "MNISTS0LS",
                "OSCIPATH",
                "BDQRTIC",
                "BRYBND",
                "LUKSAN11LS",
                "EIGENBLS",
                "MSQRTALS",
                "SPMSRTLS",
                "QING",
                "EXTROSNB",
                "WOODS",
                "TQUARTIC",
                "FREUROTH",
                "ARGLINA",
                "BROYDN3DLS",
                "LUKSAN17LS",
                "LIARWHD",
                "NONMSQRT",
                "FMINSURF",
                "SENSORS",
                "DIXMAANB",
                "SINQUAD",
                "POWER",
                "DIXMAANH",
                "QUARTC",
                "TOINTGSS",
                "EG2",
                "CURLY10",
                "DIXMAANJ",
                "DIXMAANE",
                "FMINSRF2",
                "PENALTY3",
                "ARWHEAD",
                "NCB20",
                "NCB20B",
                "FLETCBV2",
                "DIXMAANC",
                "SCHMVETT",
                "VARDIM",
                "BOXPOWER",
                "POWELLSG",
                "DIXMAANM",
                "NONDQUAR",
                "FLETBV3M",
                "COSINE",
                "DIXMAANP",
                "GENHUMPS",
                "DIXMAANF",
                "DIXMAANN",
                "CURLY20",
                "DIXMAANI",
                "DIXMAANA",
                "FLETCBV3",
                "NONCVXUN",
                "NONCVXU2",
                "DIXMAANG",
                "ENGVAL1",
                "EDENSCH",
                "DIXMAANO",
                "FLETCHCR",
                "DIXMAANK",
                "SPARSINE",
                "DQRTIC",
                "DIXMAANL",
                "BROYDN7D",
                "INDEFM",
                "BOX",
                "VAREIGVL",
                "CRAGGLVY",
                "CURLY30",
                "DIXMAAND",
                "OSBORNEA",
                "SPARSQUR",
                "CHWIRUT2LS",
                "GAUSSIAN",
                "DEVGLA2NE",
                "KOWOSB",
                "BOX3",
                "DENSCHNB",
                "CHWIRUT1LS",
                "WAYSEA2",
                "DENSCHNE",
                "HATFLDD",
                "CUBE",
                "VESUVIOULS",
                "ECKERLE4LS",
                "CERI651ELS",
                "BIGGS6",
                "HATFLDFL",
                "STREG",
                "MGH10SLS",
                "MISRA1DLS",
                "MISRA1ALS",
                "NELSONLS",
                "SNAIL",
                "DMN37143LS",
                "HYDC20LS",
                "VIBRBEAM",
                "DJTL",
                "OSCIGRAD",
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

        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, HProp_all, mypath)            
        
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
    
        
        
def run_algorithms(obj, x0, methods, algPara, HProp_all, mypath):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []            
    record_txt = lambda filename, myrecord: np.savetxt(
            os.path.join(mypath, filename+'.txt'), myrecord.cpu(), delimiter=',')  
    if 'Naive_Newton_MR' in methods:  
        print(' ')
        myMethod = 'Naive_Newton_MR'
        for i in range(len(HProp_all)):
            HProp = HProp_all[i]
            if HProp != 1:
                obj_subsampled = lambda x, control=None, index=None: obj(x, control, HProp, index)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp_all[i]*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Naive_Newton_MR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.show)
            if algPara.savetxt:
                np.savetxt(os.path.join(mypath, myMethod+'.txt'), record.cpu(), delimiter=',')
            record_all.append(record.cpu())
        
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
                    algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                    algPara.epsilon, algPara.show, record_txt)
            if algPara.savetxt:
                record_txt(myMethod, record.cpu())
            record_all.append(record.cpu())
        
    if 'L_BFGS' in methods:
        print(' ')
        myMethod = 'L_BFGS'
        print(myMethod)
        record_all.append(myMethod)
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.L, algPara.beta, 
                algPara.beta2, algPara.show, record_txt)
        if algPara.savetxt:
            np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record.cpu(), delimiter=',')
        record_all.append(record.cpu())
                    
    if 'Nonlinear_CG' in methods:
        print(' ')
        myMethod = 'Nonlinear_CG'
        print(myMethod)
        record_all.append(myMethod)
        x, record = Nonlinear_CG(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                algPara.beta3, algPara.show, record_txt)
        if algPara.savetxt:
            np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record.cpu(), delimiter=',')
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
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.beta2, algPara.epsilon, 
                    algPara.show)
            if algPara.savetxt:
                np.savetxt(os.path.join(mypath, myMethod+'.txt'), record.cpu(), delimiter=',')
            record_all.append(record.cpu())
    
    if 'Newton_CG_LS' in methods:  
        print(' ')
        myMethod = 'Newton_CG_LS'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_LS(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.epsilon, 
                    algPara.show, record_txt)
            if algPara.savetxt:
                record_txt(myMethod, record.cpu())
            record_all.append(record.cpu())
    
    if 'Newton_CG_LS_FW' in methods:  
        print(' ')
        myMethod = 'Newton_CG_LS_FW'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_LS(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, algPara.epsilon, 
                    algPara.show, record_txt, 'NC_FW')
            if algPara.savetxt:
                record_txt(myMethod, record.cpu())
            record_all.append(record.cpu())
    
            
    if 'Newton_CG_TR_Steihaug' in methods:  
        print(' ')
        myMethod = 'Newton_CG_TR_Steihaug'
        for j in range(len(HProp_all)):
            HProp = HProp_all[j]
            if HProp != 1:
                obj_subsampled = lambda x, control=None: obj(x, control, HProp)
                myMethod = 'ss' + myMethod + '_%s%%' % (int(HProp*100))
            else:
                obj_subsampled = obj
            print(myMethod)
            record_all.append(myMethod)
            x, record = Newton_CG_TR_Steihaug(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            if algPara.savetxt:
                record_txt(myMethod, record.cpu())
            record_all.append(record.cpu())
    
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
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.epsilon, 
                    algPara.Delta, algPara.Delta_max, 
                    algPara.gama1, algPara.gama2, algPara.eta,
                    algPara.show, record_txt)
            if algPara.savetxt:
                record_txt(myMethod, record.cpu())
            record_all.append(record.cpu())
            
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