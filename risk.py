import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

@click.command()
@click.option('--ytrue', 'ytrue_path', help='path for groud truth .npy files', required=True)
@click.option('--ystd', 'ystd_path', help='path for baseline supervised model predicted as .npy files', required=True)
@click.option('--ybase', 'ybase_path', help='path for KLS classifier predictions on X_i as .npy files', required=True)
@click.option('--yfinal', 'yfinal_path', help='path for KLS classifier predictions on X_j as .npy files', required=True)
@click.option('--df', 'df_path', help='path for dataframe files including the colume of [fast prog]: c_j-c_i > 1', required=True)
@click.option('--savedir', 'save_path', help='path for figure saving', default='./')
def risk_roc(
    ytrue_path: str,  
    ystd_path: str,     
    ybase_path: str,    
    yfinal_path: str,   
    df_path: str,       
    save_path: str      
):
    y_true = np.load(ytrue_path)
    y_std = np.load(ystd_path)
    y_base = np.load(ybase_path)
    y_final = np.load(yfinal_path)
    df = pd.read_csv(df_path)

    progs = [[0, 4], [1, 4], [2, 4], [0, 3], [1, 3], [0, 2]]
    sums = []
    for n in range(len(y_base)):
        pr_b = y_base[n]
        pr_f = y_final[n]
        local_sum = 0
        for m in progs:
            local_sum += pr_b[m[0]]*pr_f[m[1]]
        sums.append(local_sum)
        
    fpr, tpr, _ = roc_curve(y_true, y_std)
    fpr_, tpr_, _ = roc_curve(df['fast prog'].tolist(), sums)

    _, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 100)
    _ = ax1.plot(fpr, tpr, 'b-', alpha = 1, label = 'Supervised Model (AUC:%2.2f)' % roc_auc_score(y_true, y_std))
    _ = ax1.plot(fpr_, tpr_, 'r-', alpha = 1, label = 'Our Model (AUC:%2.2f)' % roc_auc_score(df['fast prog'].tolist(), sums))
    _ = ax1.plot(fpr, fpr, 'k--', label = 'Random Guessing')

    _ = ax1.set_xlabel('False Positive Rate')
    _ = ax1.set_ylabel('True Positive Rate')
    _ = ax1.legend(loc = 4, prop={'size': 10})
    _ = plt.savefig(save_path, dpi=600, bbox_inches = 'tight', pad_inches = 0)
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    risk_roc()
#----------------------------------------------------------------------------