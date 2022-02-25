import matplotlib.pyplot as plt
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def derivativetest(obj, x0):
    """
    Test the gradient and Hessian of a function. A large proportion 
    parallel in the middle of both plots means accuraccy.
    INPUTS:
        fun: a function handle that gives f, g, Hv
        x0: starting point
    OUTPUTS:
        derivative test plots
    """
    device = x0.device
    dt = x0.dtype
    fk, grad, Hvp = obj(x0)
    dx = torch.ones(len(x0), device = device, dtype=dt)
    M = 20;
    dxs = torch.zeros(M,1, device = device, dtype=dt)
    firsterror = torch.zeros(M, device = device, dtype=dt)
    order1 = torch.zeros(M-1, device = device, dtype=dt)
    seconderror = torch.zeros(M, device = device, dtype=dt)
    order2 = torch.zeros(M-1, device = device, dtype=dt)
    for i in range(M):
    # with torch.no_grad():
        # for param in model.parameters():
        #     param += learning_rate * param.grad
        # x = parameters_to_vector(model.parameters())
        # vector_to_parameters(x+dx, model.parameters())
        x = x0 + dx
        f2 = obj(x, 'f')
        H0 = Ax(Hvp, dx)
        # H0 = parameters_to_vector(Hv)
        firsterror[i] = abs(f2 - (fk + torch.dot(
                dx, grad)))/abs(fk)
        seconderror[i] = abs(f2 - (fk + torch.dot(
                dx, grad) + 0.5* torch.dot(dx, H0)))/abs(fk)
        print('First Order Error is %8.2e;   Second Order Error is %8.2e'% (
                firsterror[i], seconderror[i]))
        if i > 0:
            order1[i-1] = torch.log2(firsterror[i-1]/firsterror[i])
            order2[i-1] = torch.log2(seconderror[i-1]/seconderror[i])
        dxs[i] = torch.norm(dx)
        dx = dx/2
    
    step = [2**(-i-1) for i in range(M)]
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.loglog(step, abs(firsterror.cpu().detach().numpy()),'b', label = '1st Order Err')
    plt.loglog(step, (dxs.cpu().detach().numpy())**2,'r', label = 'order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(222)
    plt.semilogx(step[1:], order1.cpu().detach().numpy(),'b', label = '1st Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(223)
    plt.loglog(step, abs(seconderror.cpu().detach().numpy()),'b', label = '2nd Order Err')
    plt.loglog(step, (dxs.cpu().detach().numpy())**3,'r', label = 'Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(224)
    plt.semilogx(step[1:], order2.cpu().detach().numpy(),'b', label = '2nd Order')
    plt.gca().invert_xaxis()
    plt.legend()
            
    return plt.show()


def derivativetest_class(obj, x0):
    """
    Test the gradient and Hessian of a function. A large proportion 
    parallel in the middle of both plots means accuraccy.
    INPUTS:
        fun: a function handle that gives f, g, Hv
        x0: starting point
    OUTPUTS:
        derivative test plots
    """
    device = x0.device
    dt = x0.dtype
    fk = obj(x0).f()
    grad = obj(x0).g()
    dx = torch.ones(len(x0), device = device, dtype=dt)
    M = 20;
    dxs = torch.zeros(M,1, device = device, dtype=dt)
    firsterror = torch.zeros(M, device = device, dtype=dt)
    order1 = torch.zeros(M-1, device = device, dtype=dt)
    seconderror = torch.zeros(M, device = device, dtype=dt)
    order2 = torch.zeros(M-1, device = device, dtype=dt)
    for i in range(M):
    # with torch.no_grad():
        # for param in model.parameters():
        #     param += learning_rate * param.grad
        # x = parameters_to_vector(model.parameters())
        # vector_to_parameters(x+dx, model.parameters())
        x = x0 + dx
        f2 = obj(x).f()
        H0 = obj(x0).Hv(dx)
        # H0 = parameters_to_vector(Hv)
        firsterror[i] = abs(f2 - (fk + torch.dot(
                dx, grad)))/abs(fk)
        seconderror[i] = abs(f2 - (fk + torch.dot(
                dx, grad) + 0.5* torch.dot(dx, H0)))/abs(fk)
        print('First Order Error is %8.2e;   Second Order Error is %8.2e'% (
                firsterror[i], seconderror[i]))
        if i > 0:
            order1[i-1] = torch.log2(firsterror[i-1]/firsterror[i])
            order2[i-1] = torch.log2(seconderror[i-1]/seconderror[i])
        dxs[i] = torch.norm(dx)
        dx = dx/2
    
    step = [2**(-i-1) for i in range(M)]
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.loglog(step, abs(firsterror.cpu().detach().numpy()),'b', label = '1st Order Err')
    plt.loglog(step, (dxs.cpu().detach().numpy())**2,'r', label = 'order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(222)
    plt.semilogx(step[1:], order1.cpu().detach().numpy(),'b', label = '1st Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(223)
    plt.loglog(step, abs(seconderror.cpu().detach().numpy()),'b', label = '2nd Order Err')
    plt.loglog(step, (dxs.cpu().detach().numpy())**3,'r', label = 'Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(224)
    plt.semilogx(step[1:], order2.cpu().detach().numpy(),'b', label = '2nd Order')
    plt.gca().invert_xaxis()
    plt.legend()
            
    return plt.show()

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax =torch.mv(A, x)
    return Ax