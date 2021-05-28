# GGLasso
This package contains algorithms for solving Single and Multiple Graphical Lasso problems. Moreover, it contains the option of including latent variables.<br>

[Docs](https://gglasso.readthedocs.io/en/latest/) | [Examples](https://gglasso.readthedocs.io/en/latest/auto_examples/index.html)

## Getting started
Clone the repository, for example with

    git clone https://github.com/fabian-sp/GGLasso.git

Set up the dependencies with

    pip install -r requirements.txt

In order to install `gglasso` in your Python environment, run

    python setup.py

Test your installation with 

    pytest gglasso/ -v


### Advanced options

If you want to install dependencies with `conda`, you can run

	$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

If you wish to install `gglasso` in developer mode, i.e. not having to reinstall `gglasso` everytime you change the source code in your local repository, run

    python setup.py clean --all develop clean --all



## Algorithms
`GGLasso` contains several algorithms for Single and Multiple (i.e. Group and Fused) Graphical Lasso problems. Moreover, it allows to model latent variables (Latent variable Graphical Lasso) in order to estimate a precision matrix of for *sparse - low rank*.
<br>
1) ADMM for Group and Fused Graphical Lasso<br>
The algorithm was proposed in [2] and [3]. To use this, import `ADMM_MGL` from `gglasso/solver/admm_solver`.<br>

2) A Proximal Point method for Group and Fused Graphical Lasso<br>
We implemented the PPDNA Algorithm implemented like proposed in [4]. To use this, import `warmPPDNA` from `gglasso/solver/ppdna_solver`.<br>

3) ADMM for Single Graphical Lasso<br>

4) ADMM method for Group Graphical Lasso where the features/variables are non-conforming<br>
Method for problems where not all variables exist in all instances/datasets.  To use this, import `ext_ADMM_MGL` from `gglasso/solver/ext_admm_solver`.<br>



## References
*  [1] Friedman, J., Hastie, T., and Tibshirani, R. (2007).  Sparse inverse covariance estimation with the Graphical Lasso. Biostatistics, 9(3):432–441.
*  [2] Danaher, P., Wang, P., and Witten, D. M. (2013). The joint graphical lasso for inverse covariance estimation across multiple classes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(2):373–397.
* [3] Tomasi, F., Tozzo, V., Salzo, S., and Verri, A. (2018). Latent Variable Time-varying Network Inference. InProceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.
* [4] Zhang, Y., Zhang, N., Sun, D., and Toh, K.-C. (2020). A proximal point dual Newton algorithm for solving group graphical Lasso problems. SIAM J. Optim., 30(3):2197–2220.