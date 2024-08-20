from monteprediction.api import get_truths, get_covariance, get_mean
import numpy as np
import pandas as pd
from precise.skaters.portfoliostatic.unitport import unit_port, unit_port_p050, unit_port_p020, unit_port_p040, unit_port_p060, unit_port_p080
from cwport import cw_port
from yahooempiricalcov import yahoo_empirical_cov
from portmetrics import PORT_METRICS, total_return, sharpe_ratio, sortino_ratio

# Define the portfolio methods
portfolio_methods = [
    unit_port,
    cw_port
]


def precise_backtest(port, burn_in=8, empirical=True, metric=None):
    """
    :param port:   A function taking cov -> weights as per the convention in precise.skaters.portfoliostatic
    :param burn_in: Number of initial periods to exclude
    :return: Total return
    """
    if metric is None:
        metric = total_return

    df_truth = get_truths()
    expiries = df_truth.index.values
    port_rets = []

    for expiry in expiries[burn_in:]:
        print(expiry)
        df_mu = get_mean(expiry=expiry)

        emp_cov = yahoo_empirical_cov(expiry=expiry)
        community_cov = get_covariance(expiry=expiry).values
        if np.any(pd.isnull(emp_cov)):
            raise ValueError('shit')

        if empirical:
            cov = emp_cov
        else:
            cov = community_cov

        if np.issubdtype(community_cov.dtype, np.number):
            if not np.any(np.isnan(community_cov)) and not np.any(pd.isnull(community_cov)):
                mu = df_mu.values
                truth = df_truth.loc[expiry].values
                w = port(cov=cov)
                port_ret = np.log(np.dot(w, np.exp(truth)))
                if np.isnan(port_ret):
                    raise ValueError('port return is nan')
                port_rets.append(port_ret)

    score = metric(port_rets)

    return score


def metric_leaderboard(portfolio_methods, metric, empirical=False):
    results = {}
    for port in portfolio_methods:
        print(f"Running backtest for {port.__name__}")
        total_ret = precise_backtest(port=port, metric=metric, empirical=empirical)
        results[port.__name__] = total_ret

    # Sort the results by total return from highest to lowest
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    return sorted_results


def community_covariance_comparison(portfolio_methods, metric):
    print(f"Comparing metric: {metric.__name__}")

    empirical_results = metric_leaderboard(portfolio_methods, metric, empirical=True)
    non_empirical_results = metric_leaderboard(portfolio_methods, metric, empirical=False)

    # Create a DataFrame to align results for comparison
    all_methods = list(set(empirical_results.keys()).union(set(non_empirical_results.keys())))
    comparison_df = pd.DataFrame(index=all_methods, columns=['Empirical cov', 'Community cov'])

    for method in all_methods:
        comparison_df.loc[method, 'Empirical cov'] = empirical_results.get(method, 'N/A')
        comparison_df.loc[method, 'Community cov'] = non_empirical_results.get(method, 'N/A')

    # Add column to indicate which method is better
    comparison_df['Community helps'] = comparison_df.apply(lambda row: 1 if row['Community cov'] > row['Empirical cov'] else 0,
                                                  axis=1)

    print(f"Comparison of {metric.__name__} using Empirical vs. Community Covariance:")
    print(comparison_df)
    return comparison_df


def show_leaderboard():
    metric= sharpe_ratio
    leaderboard = metric_leaderboard(portfolio_methods, empirical=True, metric = metric)
    print("Leaderboard: "+metric.__name__)
    for method, total_ret in leaderboard.items():
        print(f"{method}: {total_ret:.4f}")


if __name__=='__main__':
    metric = sharpe_ratio
    community_covariance_comparison(portfolio_methods, metric)


