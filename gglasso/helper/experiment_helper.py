import numpy as np

def lambda_parametrizer(w1 = 0.1, w2 = 0.2):
    
    l2 = np.sqrt(2) * w1 * w2
    l1 = w1 - (1/np.sqrt(2)) * l2
   
    return l1,l2

