from monteprediction.api import get_covariance, get_mean, get_truths
import numpy as np
import pandas as pd
from precise.skaters.portfoliostatic.unitport import unit_port as port

# Demonstrates how to use community covariance to create and index alternative


def precise_backtest(port, burn_in=4):
    """
    :param port:   A function taking cov -> weights as per the convention in precise.skaters.portfoliostatic
    :return:
    """
    df_truth = get_truths()
    expiries = df_truth.index.values
    port_rets = list()
    for expiry in expiries[burn_in:]:
        print(expiry)
        df_mu = get_mean(expiry=expiry)
        df_cov = get_covariance(expiry=expiry)
        cov = df_cov.values
        if np.issubdtype(cov.dtype, np.number):
           if not np.any(np.isnan(cov)) and not np.any(pd.isnull(cov)):
                mu = df_mu.values
                truth = df_truth.loc[expiry].values
                w = port(cov=cov)
                port_ret = np.log(np.dot(w, np.exp(truth)))
                port_rets.append(port_ret)

    total_ret = np.exp(np.sum(port_rets))-1
    print({'port_ret_bps':10000*total_ret})
    return total_ret


if __name__=='__main__':
    precise_backtest(port)



