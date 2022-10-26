"""
Code for "A Newton-MR algorithm with complexity guarantees for nonconvex
smooth unconstrained optimization"
Authors: Yang Liu and Fred Roosta. ArXiv: 2208.07095
"""

from initialize import initialize
import torch
        
class algPara():
    def __init__(self, value):
        self.value = value
        
class nsPara():
    def __init__(self, value):
        self.value = value


#initialize methods
data = [
        # 'real-sim',
        # 'rcv1_train'
        # 'mnist',
        'cifar10',
        # 'gisette', 
        # '20news', 
        # 'stl10',
        # 'arcene',
        # 'hapt',
        # 'power_plant',
        # 'housing',    
        ]

prob = [
#        'logitreg', # realData, pytorch support
        # 'softmax', # realData, pytorch 
        # 'student_t',
        'Auto-encoder', # pytorch support #float32
        # 'nls',
        ]

methods = [
            # 'Naive_Newton_MR',
            'Newton_MR', # alpha0 = 1, rho = 0.25, + NC forward linesearch
            'L_BFGS',
            # "Nonlinear_CG",
            'Newton_CG_LS', # Royer
            'Newton_CG_LS_FW', # Royer
            'Newton_CR', # Dahito
            'Newton_CG_TR_Steihaug', # Steihaug CG TR
            'Newton_CG_TR_Pert', # Curtis CG TR
        ]

regType = [
                # 'None',
                # 'Convex',
        'Nonconvex',
        ] 

#initial point
x0Type = [
                    'randn',
            # 'rand',
        # 'ones'
            # 'zeros', # note: 0 is a saddle point for fraction problems
        ]

#initialize parameter
algPara.funcEvalMax = 1e5 #Set mainloop stops with Maximum Function Evaluations
algPara.mainLoopMaxItrs = 1E5 #Set mainloop stops with Maximum Iterations
algPara.innerSolverMaxItrs = 1E3
algPara.lineSearchMaxItrs = 1E3
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.epsilon = 1E-5
algPara.innerSolverTol = 1E-1 #relative residual for inner solver MR
algPara.beta = 1E-4 #Armijo Line search para
algPara.beta2 = 0.9 #Wolfe's condition for L-BFGS
algPara.beta3 = 0.1 #Wolfe's condition for Nonlinear_CG
algPara.L = 10 # L-BFGS limited memory
algPara.Delta = 1E0
algPara.Delta_min = 1E-6
algPara.Delta_max = 1E10 #Trust-region redials max
algPara.gama1 = 3 # enlarge
algPara.gama2 = 0.5 # shrink
algPara.show = True
algPara.savetxt = True
algPara.reorth = True # reorthogonalization inside MINRES
algPara.cuda = True
algPara.cuda = False
lamda = 1E-3 #regularizer
algPara.zoom = 1E8
algPara.student_t_nu = 1
algPara.eta = 0.2 # < 0.25 in Trust-Region Newton-CG
# algPara.cutData = 1000
# algPara.normalizex0 = True
algPara.normalizex0 = False
algPara.dType = torch.float64

##########################For 1 run plot#####################################
# comment off HProp if you don't want subsampling
#HProp_all = [0.1, 0.05, 0.01] # \leq 3 inputs
#HProp_all = [0.05]
HProp_all = [1] # full Hessian
plotAll = False
#########################GMM performance profile plot########################
# multiple run performance profile plot 
#algPara.total_run = 1 # Cutest multiple problem select

## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, 
               algPara, plotAll, lamda)