
"""
This is Condat's algorithm
y is the point where we calculate prox_lambda*||By||_1

"""

import numpy as np
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
                x[k0:kminus +1] = vmin
                k=k0=kminus = kminus + 1
                vmin = y[k] ; umax = y[k] + lam - vmax; umin = lam
            elif umax > 0:
                #print('c2')
                x[k0:kminus +1] = vmax
                k=k0=kplus = kplus + 1
                umin = y[k] - lam - vmin; vmax = y[k] ; umax = -lam
            else:
                #print('c3')
                x[k0:N+1] = vmin + umin /(k-k0+1)
                return x
        
        if y[k+1] + umin - vmin < -lam:
            #print('b1')
            x[k0:kminus+1] = vmin
            k = k0 = kplus = kminus = kminus + 1
            vmin = y[k]; vmax = y[k] + 2*lam
            umin = lam; umax = -lam
    
        elif y[k+1] + umax -vmax > lam:
            #print('b2')
            x[k0:kplus+1] = vmax
            k = k0 = kplus = kminus = kplus + 1
            vmin = y[k] - 2*lam; vmax = y[k]
            umin = lam; umax = -lam
            
        else:
            #print('b3')
            k += 1
            umin = umin + y[k] - vmin
            umax = umax + y[k] - vmax
            if umin >= lam:
                #print('b31')
                vmin = vmin + (umin -lam)/(k-k0+1); umin = lam; kminus = k
            elif umax <= -lam:
                #print('b32')
                vmax = vmax + (umax +lam)/(k-k0+1); umax = -lam; kplus = k

        
#%%


for i in range(1000):
    print(i)
    y = np.random.rand(10)
    x = condat_method(y, 1)
