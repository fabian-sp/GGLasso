"""
Tests that the SpiecEasi way to fix the rank of L can be enforced through the glasso_problem class.

TODO: assert on correct rank recovery under appropriate data?
"""
import numpy as np
from gglasso.problem import glasso_problem
from gglasso.helper.basic_linalg import scale_array_by_diagonal

############ Generate data
p = 30
N = 10000

# generate Theta
A = np.random.randn(p,p)
Theta = A.T@A + 150*np.eye(p)

scale = np.tile(np.sqrt(np.diag(Theta)),(p,1))
scale = scale.T * scale
        
Theta = Theta / scale
Theta[np.abs(Theta) <= 0.05] = 0

assert np.all(np.linalg.eigvalsh(Theta) > 0)
print("smallest eigval of Theta:", np.linalg.eigvalsh(Theta).min())

# generate L
L = np.zeros((p,p))
rank = 5
for i in range(rank):
    v = np.random.rand(p)
    L += np.outer(v,v)

L = L *(0.6 / np.linalg.eigvalsh(L).max()) 
print("Rank of L:", np.linalg.matrix_rank(L))

R = Theta - L
print("smallest eigval of R:", np.linalg.eigvalsh(R).min())
assert np.all(np.linalg.eigvalsh(R) > 0)

# generate S
Sigma = np.linalg.pinv(R)

sample = np.zeros((p, N))
sample = np.random.multivariate_normal(np.zeros(p), Sigma, N).T

S = np.cov(sample, bias=True)
S = scale_array_by_diagonal(S)

D = np.linalg.eigvalsh(S)
print("Eigvals of S:", np.linalg.eigvalsh(S))


############ Solve
P = glasso_problem(S=S, N=N, reg=None, latent=True)
print(P)

# To match SpiecEasi SLR
modelselect_params = {
    "lambda1_range": np.logspace(0, -3, 10),
    "mu1_range": np.linspace(5, 1, 5),
    "off_diagonal_l1": False,
    "fix_latent_rank": True
}

P.model_selection(modelselect_params=modelselect_params, method='eBIC', gamma=0.1)
print(P.reg_params)
stats = P.modelselect_stats
