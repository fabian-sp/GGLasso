# GGLasso
This package contains implementations for the Multiple Graphical Lasso problem.<br>

1) Proximal Point methods for Group and Fused Graphical Lasso<br>
This is the PPDNA Algorithm implemented like proposed [here](https://arxiv.org/abs/1906.04647). To use this, import `warmPPDNA` from `gglasso/solver/ppdna_solver`.<br>

2) ADMM methods for Group and Fused Graphical Lasso<br>
Algorithm like proposed [here](https://arxiv.org/abs/1111.0324) with minor adaptions. To use this, import `ADMM_MGL` from `gglasso/solver/admm_solver`.<br>

4) ADMM method for Group Graphical Lasso where the features/variables are non-conforming<br>
 Method for problems where not all variables exist in all instances/datasets.  To use this, import `ext_ADMM_MGL` from `gglasso/solver/ext_admm_solver`.<br>


Experiments for Group and Fused GL recovery rates are in `exp_powerlaw_ggl` `exp_powerlaw_fgl`. Experiments for runtime comparison between ADMM and PPDNA are in `exp_runtime_ggl`.<br>
In order to run these, open the files in a Python IDE (e.g. Spyder) and set the working directory to `...\GGLasso` (i.e. the directory where the scripts are located).
Make sure that all packages are installed or run

    pip install -r requirements.txt

alternatively.