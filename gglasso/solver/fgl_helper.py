"""
author: Fabian Schaipp

This is Condat's algorithm for the proximal operator of the Fused Lasso regularizer (https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf)
"""

import numpy as np
from numba import jit
#from tick.prox import ProxTV

@jit(nopython = True)
def condat_method(y,lam):

    N = len(y)
    
    x = np.zeros(N)
    k = k0 = kplus = kminus = 0
    
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    
    while 0 < 1:
        
        while k == N-1:
            if umin < 0:
                x[k0:kminus +1] = vmin
                kminus += 1
                k = k0 = kminus
                umin = lam; vmin = y[k]; umax = y[k] + lam - vmax 
            elif umax > 0:
                x[k0:kplus+1] = vmax
                kplus += 1
                k  = k0 = kplus
                umax = -lam; vmax = y[k]; umin = y[k] - lam - vmin
            else:
                x[k0:] = vmin + umin/(k-k0+1)
                return x
            
            if k == N-1:
                x[k] = vmin + umin
                return x
    
        if y[k+1] + umin - vmin < -lam:
            x[k0:kminus+1] = vmin
            kminus += 1
            k = kplus = k0 = kminus
            vmin = y[k]; vmax = y[k] + 2*lam
            umin = lam; umax = -lam
        
        elif y[k+1] + umax -vmax > lam:         
            x[k0:kplus+1] = vmax
            kplus += 1
            k  = kminus = k0 = kplus
            vmin = y[k] - 2*lam; vmax = y[k]
            umin = lam; umax = -lam
                     
        else:
            k += 1
            umin = umin + y[k] - vmin
            umax = umax + y[k] - vmax
            if umin >= lam:
                vmin += (umin-lam)/(k-k0+1); umin = lam; kminus = k
            if umax <= -lam:
                vmax += (umax+lam)/(k-k0+1); umax = -lam; kplus = k
         
    return 

# for testing

# def objective(x, y, l1):
#     res = l1* abs(x[1:] - x[0:-1]).sum() + .5*np.linalg.norm(x-y)**2
    
#     return res


#l1 = .1
#
# for i in range(1000):
#     print(i)
#     y = np.random.rand(100)
#     x = condat_method(y, l1)
#     x2 = ProxTV(l1).call(y)
    
#     print(np.linalg.norm(x-x2))
#     print("Condat function correct: ", objective(x, y, l1) <= objective(x2, y, l1) + 1e-5)
#     assert  np.linalg.norm(x-x2) <= 1e-10
    


    
    




