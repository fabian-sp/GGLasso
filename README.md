# GGLasso

[![PyPI version fury.io](https://badge.fury.io/py/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)
[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/gglasso/badge/?version=latest)](http://gglasso.readthedocs.io/?badge=latest)


This package contains algorithms for solving General Graphical Lasso (GGLasso) problems, including single, multiple, as well as latent 
Graphical Lasso problems. <br>

[Docs](https://gglasso.readthedocs.io/en/latest/) | [Examples](https://gglasso.readthedocs.io/en/latest/auto_examples/index.html)

## Getting started

### Install via pip

The package is available on pip and can be installed with

    pip install gglasso

### Install from source

Alternatively, you can install the package from source using the following commands:

    git clone https://github.com/fabian-sp/GGLasso.git
    pip install -r requirements.txt
    python setup.py

Test your installation with 

    pytest gglasso/ -v


### Advanced options

When installing from source, you can also install dependencies with `conda` via the command

	$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

If you wish to install `gglasso` in developer mode, i.e. not having to reinstall `gglasso` everytime the source code changes (either by remote or local changes), run

    python setup.py clean --all develop clean --all

## The `glasso_problem` class

`GGLasso` can solve multiple problem forumulations, e.g. single and multiple Graphical Lasso problems as well as with and without latent factors. Therefore, the main entry point for the user is the `glasso_problem` class which chooses automatically the correct solver and model selection functionality. See [our documentation](https://gglasso.readthedocs.io/en/latest/problem-object.html) for all the details.


## Algorithms

`GGLasso` contains algorithms for Single and Multiple Graphical Lasso problems. Moreover, it allows to model latent variables (Latent variable Graphical Lasso) in order to estimate a precision matrix of type **sparse - low rank**. The following algorithms are contained in the package.
<br>
1) ADMM for Single Graphical Lasso<br>

2) ADMM for Group and Fused Graphical Lasso<br>
The algorithm was proposed in [2] and [3]. To use this, import `ADMM_MGL` from `gglasso/solver/admm_solver`.<br>

3) A Proximal Point method for Group and Fused Graphical Lasso<br>
We implement the PPDNA Algorithm like proposed in [4]. To use this, import `warmPPDNA` from `gglasso/solver/ppdna_solver`.<br>

4) ADMM method for Group Graphical Lasso where the features/variables are non-conforming<br>
Method for problems where not all variables exist in all instances/datasets.  To use this, import `ext_ADMM_MGL` from `gglasso/solver/ext_admm_solver`.<br>

<!---
## Benchmarking

We solve each of the problems with each solver for different tolerance values, and we compare the obtained solutions to a reference solution, denoted by <img src="https://render.githubusercontent.com/render/math?math=Z%5E%5Cast%0A">.
In the figure below it is obtained by solving a SGL problem by one of the solvers for very small tolerance values (we used [regain](https://github.com/fdtomasi/regain) and set tol=rtol=1e-10). Finally, for a solution <img src="https://render.githubusercontent.com/render/math?math=Z%0A">, we calculate its accuracy using the normalized Euclidean distance:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Ctext%7Baccuracy%7D(Z)%20%3D%20%20%5Cfrac%7B%5C%7CZ%5E%5Cast%20-%20Z%20%5C%7C%7D%7B%20%5C%7C%20Z%5E%5Cast%5C%7C%20%7D">
</p>

Now, we select for each solver the run with minimal runtime where accuracy of Z is less (or equal) to a maximal accuracy of our choice (e.g., 0.005).

![Accuracy of 0.005](https://github.com/fabian-sp/GGLasso/blob/f-joss-paper/benchmarks/bm_accuracy_0.005.png)
--->


## Community Guidelines

1)  Contributions and suggestions to the software are always welcome.
    Please, consult our [contribution guidelines](CONTRIBUTING.md) prior
    to submitting a pull request.
2)  Report issues or problems with the software using github’s [issue
    tracker](https://github.com/fabian-sp/GGLasso/issues).
3)  Contributors must adhere to the [Code of
    Conduct](CODE_OF_CONDUCT.md).


## References
*  [1] Friedman, J., Hastie, T., and Tibshirani, R. (2007).  Sparse inverse covariance estimation with the Graphical Lasso. Biostatistics, 9(3):432–441.
*  [2] Danaher, P., Wang, P., and Witten, D. M. (2013). The joint graphical lasso for inverse covariance estimation across multiple classes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(2):373–397.
* [3] Tomasi, F., Tozzo, V., Salzo, S., and Verri, A. (2018). Latent Variable Time-varying Network Inference. InProceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.
* [4] Zhang, Y., Zhang, N., Sun, D., and Toh, K.-C. (2020). A proximal point dual Newton algorithm for solving group graphical Lasso problems. SIAM J. Optim., 30(3):2197–2220.
