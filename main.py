# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:20:01 2021

@author: Liu Yang
"""

from initialize import initialize
import torch

class learningRate():
    def __init__(self, value):
        self.value = value
        
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
        'mnist',
#          'cifar10',
            # 'gisette', 
          # 'arcene',
#         'hapt',
        # 'power_plant',
#         'arcene',
        # 'housing',       

#CUTEst problems import any type of problems
        # 'import_cutest',
#        'UR',
        # 'UO',
#        'U',
        # 'EG2'
        # 'PALMER1D',
        # 'PALMER1C',
        # 'PALMER1D',
#        'BROWNAL',
#        'EDENSCH',
        # 'PALMER1C'
        # 'PALMER3C',
#        'PALMER4C',
#        'PALMER5C',
        # 'PALMER6C',
        # 'SPINLS',
        # 'PARKCH',
        # 'STRATEC',
        # 'SPINLS',
#        'WALL20',
#        'WALL50',
#        'WALL100',
        # 'GULF',
        
        #good in gradient, bad in function value
        # 'MEYER3',
        
        #bad results        
        # 'ERRINROS',
        # 'EXTROSNB',
        # 'STREG',
        
        #Nan
#        'BA-L1LS',
#        'BA-L21LS',
        # 'BARD',
        # 'BDQRTIC',
# =============================================================================
#         'BEALE', #!
# =============================================================================
#        'BIGGS6', #very bad in Newton-MR#
        # 'BOX',
#        'BOX3',
#        'CKOEHELB', #error#
#        'CRAGGLVY',
#        'DIXMAANN',
#        'BA-L1LS',
        ]

prob = [
#        'logitreg', # realData, pytorch support
        # 'softmax', # realData, pytorch 
#        'student_t',
#         'Auto-encoder', # pytorch support #float32
              'nls',
##########################################################
#            'cutest',
        ]

methods = [
        'Newton_MR_invex',
        'Newton_MR', # alpha0 = 1, rho = 0.25, + NC forward linesearch
########################################################
        # 'L_BFGS',
        # 'Newton_CG', # not for least_square
        # 'Gauss_Newton', # not for softmax
##########################################################
        
        # 'Momentum',
        # 'Adagrad',
        # 'Adadelta',
        # 'RMSprop',
        # 'Adam',
        # 'SGD',
##########################################################
        # 'Newton_MR_TR_NS', ## ['TR', 'TR_NS']      
        
          # 'Damped_Newton_CG', # Royer
        
#        'Newton_CR', # Dahito
#         'Newton_CG_TR', # Steihaug CG TR
#           'Newton_CG_TR_Pert', # Curtis CG TR  
########################################################

        ]

algPara.regType = [
#               'None',
        'Convex',
#         'Nonconvex',
#         'Nonsmooth',
        ] 

#initial point
x0Type = [
                 'randn',
#            'rand',
#         'ones'
#             'zeros', # note: 0 is a saddle point for fraction problems
        ]
algPara.seed = 0
algPara.nperseed = 1
#initialize parameter
algPara.funcEvalMax = 1E3 #Set mainloop stops with Maximum Function Evaluations
algPara.mainLoopMaxItrs = 1E6 #Set mainloop stops with Maximum Iterations
algPara.innerSolverMaxItrs = 1E4
algPara.lineSearchMaxItrs = 1E3
algPara.psi = 1e-12
algPara.gradTol = 1e-7 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 1E-6 #relative residual for inner solver MR
algPara.innerSolverTol1 = algPara.innerSolverTol #relative residual for inner solver CG
algPara.innerSolverTol2 = 1E-6 #residual vs. norm_y of inner solver for Perturbations
#algPara.innerSolverTol = 'exact' #Inexactness of inner solver
algPara.beta = 1E-4 #Armijo Line search para
algPara.beta0 = 0.25 #Line search para only for Newton-MR!
#algPara.beta0 = 1E-4 #Line search para only for Newton-MR!
# algPara.beta = 0.25 #Line search para
algPara.beta2 = 0.9 #Wolfe's condition for L-BFGS
algPara.L = 5
algPara.Delta = 1E2
algPara.Delta_min = 1E-6
algPara.Delta_max = 1E25 #Trust-region redials max
algPara.gama1 = 7
algPara.gama2 = 0.01
#algPara.eta = 0.1
algPara.epsilon = 1E-10
algPara.show = True 
algPara.reorth = True
lamda = 1E-3 #regularizer
algPara.zoom = 1E0
algPara.delta = 1E1
algPara.gama = 10
algPara.student_t_nu = 5
algPara.eta = 0.2 # < 0.25 in Trust-Region Newton-CG
# algPara.cutData = 1000
algPara.dType = torch.float64
#algPara.dim = 10

#algPara.dType = torch.float64
#mydtype = torch.float64 
algPara.Quadradic_exact = True

# for softmax MNIST
# learningRate.Momentum = 1E-5
# learningRate.Adagrad = 1E-4
# learningRate.Adadelta = 1E-1
# learningRate.RMSprop = 1E-5
# learningRate.Adam = 1E-5
# learningRate.SGD = 1E-5

# for softmax Cifar10
learningRate.Momentum = 1E-3
learningRate.Adagrad = 1E-4
learningRate.Adadelta = 1E1
learningRate.RMSprop = 1E-2 #1E-6
learningRate.Adam = 1E-3 #1E-7
learningRate.SGD = 1E-3

# for gmm
#learningRate.Momentum = 1E-8
#learningRate.Adagrad = 1E-1
#learningRate.Adadelta = 10
#learningRate.RMSprop = 1E-2
#learningRate.Adam = 1E-2
#learningRate.SGD = 1E-8
##########################For 1 run plot#####################################
batchsize = 1 # proportion of mini-batch size

##########################For 1 run plot#####################################
# comment off HProp if you don't want subsampling
#HProp_all = [0.1, 0.05, 0.01] # \leq 3 inputs
#HProp_all = [0.05]
HProp_all = [1] # full Hessian
#HProp_all = [1E-2, 1E-5, 1E-13] # only for Fraction problems
plotAll = False
#########################GMM performance profile plot########################
# multiple run performance profile plot 
#algPara.total_run = 1 # Cutest multiple problem select

## Initialize
if hasattr(algPara, 'seed'):
    initialize(data, methods, prob, x0Type, HProp_all, batchsize, 
               algPara, learningRate, plotAll, lamda, 
               algPara.seed)
else:
    initialize(data, methods, prob, x0Type, HProp_all, batchsize, 
               algPara, learningRate, plotAll, lamda)

