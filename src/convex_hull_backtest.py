from monteprediction.api import get_covariance, get_mean, get_truths
import numpy as np
import pandas as pd
from cwport import cw_port
from precise.skaters.portfoliostatic.unitport import unit_port
from precise.skaters.portfoliostatic.equalport import equal_long_port
from precise.skaters.portfoliostatic.weakport import weak_h400_long_port
from precise.skaters.managers.covmanagerfactory import closest_random_nudge
from precise.skaters.covarianceutil.covrandom import jiggle_cov, DEFAULT_COV_NOISE

# Demonstrates how to use community covariance to create and index alternative


def slightly_random_unit_port(cov, noise=10*DEFAULT_COV_NOISE):
    jiggled_cov = jiggle_cov(cov=cov, noise=noise)
    return unit_port(cov=jiggled_cov)

def slightly_random_weak_port(cov, noise=10*DEFAULT_COV_NOISE):
    jiggled_cov = jiggle_cov(cov=cov, noise=noise)
    return weak_h400_long_port(cov=jiggled_cov)


def convex_hull_backtest(port, l:int=51, burn_in=4, q=0.75, lmbd=0.9):
    """
    :param port:    function taking cov -> weights
    :param lmbd:    coefficient used to smooth cov estimates (lmbd=1 means just use this week)
    :param q:       coefficient determining how far to move towards the nearest portfolio
    :param l:       number of portfolios to generate, and also determines whether to move to closest point in convex hull or just to nearest portfolio
                                      if l is odd --> use convex hull
                                      if l is even --> just use closest point
    :param burn_in: number of weeks to ignore at the start
    :return:
    """
    df_truth = get_truths()
    expiries = df_truth.index.values
    port_rets = list()
    prev_w = None
    prev_cov = None
    # Loop over weeks ...
    for expiry in expiries[burn_in:]:
        df_mu = get_mean(expiry=expiry)
        df_cov = get_covariance(expiry=expiry)

        # Get community cov and default to previous week if there are nan
        cov = df_cov.values
        if not (np.issubdtype(cov.dtype, np.number)) or (np.any(np.isnan(cov)) or np.any(pd.isnull(cov))):
            if prev_cov is not None:
                cov = np.copy(prev_cov)
            else:
                cov = np.eye(len(df_mu))

        # Smooth
        if prev_cov is not None:
            cov = lmbd*cov + (1-lmbd)*prev_cov
        prev_cov = np.copy(cov)

        if np.issubdtype(cov.dtype, np.number):
           if not np.any(np.isnan(cov)) and not np.any(pd.isnull(cov)):
                mu = df_mu.values
                truth = df_truth.loc[expiry].values

                # TODO: (optional) use mu to shrink covariances a little

                # If there is no previous target use cap weight
                if prev_w is None:
                    prev_w = cw_port(cov=cov)

                # Run portfolio construction several times and move towards the nearest point in convex hull
                w = closest_random_nudge(port, cov, q=q, l=l, w=prev_w, port_kwargs={})

                port_ret = np.log(np.dot(w, np.exp(truth)))
                port_rets.append(port_ret)

    total_ret = np.exp(np.sum(port_rets))-1
    return total_ret


if __name__=='__main__':
    burn_in = 10
    cw = convex_hull_backtest(port=cw_port, burn_in=burn_in)
    eq = convex_hull_backtest(port=equal_long_port, burn_in=burn_in)
    rup_minus_eqs = list()
    rup_minus_cws = list()
    rups = list()
    eqs = list()
    cws = list()
    for k in range(10):
        rup = convex_hull_backtest(port=slightly_random_weak_port)
        print({'rup minus cw':rup-cw,'rup minus eq':rup-eq, 'cw':cw,'eq':eq,'rup':rup})
        rups.append(rup)
        eqs.append(eq)
        cws.append(cw)








