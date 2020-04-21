# GGLasso
This package contains implementations for the Multiple Graphical Lasso problem.<br>

## Functionalities
1) Proximal Point methods for Group and Fused Graphical Lasso<br>
This is the PPDNA Algorithm implemented like proposed [here](https://arxiv.org/abs/1906.04647). To use this, import `warmPPDNA` from `gglasso/solver/ppdna_solver`.<br>

2) ADMM methods for Group and Fused Graphical Lasso<br>
Algorithm like proposed [here](https://arxiv.org/abs/1111.0324) with minor adaptions. To use this, import `ADMM_MGL` from `gglasso/solver/admm_solver`.<br>

4) ADMM method for Group Graphical Lasso where the features/variables are non-conforming<br>
 Method for problems where not all variables exist in all instances/datasets.  To use this, import `ext_ADMM_MGL` from `gglasso/solver/ext_admm_solver`.<br>

5) Model selection via grid search <br>
Method for choosing the best regularization parameters `lambda1` and `lambda2` via choosing the minimal eBIC or AIC. To use this, import `grid_search` from `gglasso/helper/model_selection`<br>

## Setup and Experiments
Make sure that all packages are installed or run

    pip install -r requirements.txt

alternatively.<br>
If you want to install with `conda`, you can run

	$ while read requirement; do conda install --yes $requirement; done < requirements.txt

In order to install a local version which you want to edit, run

    python setup.py clean --all develop clean --all

For testing and in order to understand how to call the solvers, run `example.py`

Experiments for Group and Fused GL recovery rates are in `experiments/exp_powerlaw_ggl` `experiments/exp_powerlaw_fgl`. Experiments for runtime comparison between ADMM and PPDNA are in `experiments/exp_runtime_ggl`. They use synthetic data and produce the figures shown in the thesis.<br>
In order to run these and save the figures, set the working directory to the directory where the scripts are located.
