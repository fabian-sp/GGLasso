import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_bm(df=pd.DataFrame(), lambda_list=list, min_acc=1e-2, log_scale=True):
    
    col_dict = {"gglasso": '#264F73', "gglasso-block": "#3F7EA6", 'regain': '#F2811D', 'sklearn': '#C0C0C0'}
    
    fig, axs = plt.subplots(len(lambda_list), 1, figsize=(6,10))
    j = 0
    for l1 in lambda_list:
        ax = axs[j]
        df_sub = df[(df.l1 == l1) & (df.accuracy <= min_acc)]
        tmp = df_sub.groupby(["p", "N", "method"])["time"].min()

        tmp.unstack().plot(ls='-', marker='o', xlabel="(p,N)", ylabel="runtime [sec]", ax=ax, color = col_dict)
        
        ax.set_title(rf"$\lambda_1$ = {l1}")
        ax.grid(linestyle='--')
        ax.legend(loc='upper left')
        
        if log_scale:
            ax.set_yscale('log')
            # ax.set_xscale('log')

        j += 1

    fig.tight_layout()
    return