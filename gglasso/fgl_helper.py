
"""
This is Condat's algorithm
y is the point where we calculate prox_lambda*||By||_1

"""

import numpy as np
from numba import jit
from tick.prox import ProxTV


#%%
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
                #print('c1')
                while k0 <= kminus:
                    x[k0] = vmin
                    k0+=1
                
                #x[k0:kminus +1] = vmin
                #kminus += 1
                kminus=k=k0
                umin = lam; vmin = y[k]; umax = y[k] + lam - vmax 
            elif umax > 0:
                #print('c2')
                while k0 <= kplus:
                    x[k0] = vmax
                    k0+=1
                    
                #x[k0:kplus+1] = vmax
                #kplus += 1 
                kplus=k=k0
                umax = -lam; vmax = y[k]; umin = y[k] - lam - vmin
            else:
                #print('c3')
                vmin += umin/(k-k0+1)
                x[k0:] = vmin 
                return x
        
        if y[k+1] + umin - vmin < -lam:
            #print('b1')
            while k0 <= kminus:
                x[k0] = vmin
                k0+=1
            #x[k0:kminus+1] = vmin
            #kminus += 1
            
            k = kplus = kminus = k0
            vmin = y[k]; vmax = y[k] + 2*lam
            umin = lam; umax = -lam
    
        elif y[k+1] + umax -vmax > lam:
            #print('b2')
            while k0 <= kplus:
                x[k0] = vmax
                k0+=1
            #x[k0:kplus+1] = vmax
            #kplus += 1
            
            k = kplus = kminus = k0
            vmin = y[k] - 2*lam; vmax = y[k]
            umin = lam; umax = -lam
            
        else:
            #print('b3')
            k += 1
            umin = umin + y[k] - vmin
            umax = umax + y[k] - vmax
            if umin >= lam:
                #print('b31')
                vmin += (umin-lam)/(k-k0+1); umin = lam; kminus = k
            elif umax <= -lam:
                #print('b32')
                vmax += (umax+lam)/(k-k0+1); umax = -lam; kplus = k

    
#%%

for i in range(1000):
    print(i)
    y = np.random.rand(100)
    x = condat_method(y, 1)
    x2 = ProxTV(1).call(y)
    
    print(np.linalg.norm(x-x2))
    
    
    #assert (abs(x-x2).sum() <= 1e-5)
    

    
    




