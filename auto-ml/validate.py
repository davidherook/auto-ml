import json
import argparse 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

y_pred_col_name = 'prediction'
y_true_col_name = 'lbl_acct_pm06_mmax_amt'

def mean_abs_pct_error(y_true, y_pred):
    return np.mean( np.abs( (y_true - y_pred) / y_true ) ) * 100

def validation_summary(y_true, y_pred, print_results=False, description=None):

    target_mean, target_std = np.mean(y_true), np.std(y_true)

    tst_R = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    maep = mean_abs_pct_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    err_std = (y_true - y_pred).std()

    metrics = {
        "description": description,
        "test_size": y_true.shape[0],
        "target_mean": target_mean,
        "target_std": target_std,
        "validation": {
            "Test R-sq":tst_R,
            "MAE": mae,
            "MAEP": maep,
            "RMSE": rmse,
            "Error Std": err_std
        }
    }

    if print_results:
        print(json.dumps(metrics, indent=4))

    return metrics