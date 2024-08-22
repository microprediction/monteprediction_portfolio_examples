from monteprediction.api import get_truths, get_covariance, get_mean
import numpy as np
import pandas as pd
from precise.skaters.portfoliostatic.unitport import unit_port, unit_port_p050
from cwport import cw_port
from yahooempiricalcov import yahoo_empirical_cov
from precise.skaters.portfoliostatic.equalport import equal_long_port
from precise.skaters.portfoliostatic.weakport import weak_long_port, weak_h150_long_port, weak_h400_long_port
from portmetrics import PORT_METRICS, total_return, sharpe_ratio, sortino_ratio
from covmetrics import COV_METRICS, cov_likelihood, subspace_likelihood, projected_likelihood, projected_subspace_likelihood

# Define the portfolio methods
portfolio_methods = [
    unit_port,
    unit_port_p050,
    weak_long_port,
    weak_h150_long_port,
    weak_h400_long_port,
    cw_port,
    equal_long_port
]


def precise_backtest(port, burn_in=6, empirical=True, port_metric=None, cov_metric=None, cov_metric_kwargs=None):
    """
    :param port:   A function taking cov -> weights as per the convention in precise.skaters.portfoliostatic
    :param burn_in: Number of initial periods to exclude
    :return: Total return
    """
    if cov_metric_kwargs is None:
        cov_metric_kwargs = {}

    if port_metric is None:
        port_metric = total_return

    if cov_metric is None:
        cov_metric = cov_likelihood

    df_truth = get_truths()
    expiries = df_truth.index.values
    port_rets = []
    cov_mets = []

    for expiry in expiries[burn_in:]:
        print(expiry)
        df_mu = get_mean(expiry=expiry)
        emp_cov = yahoo_empirical_cov(expiry=expiry)
        community_cov = get_covariance(expiry=expiry).values
        if np.any(pd.isnull(emp_cov)):
            raise ValueError('shit')

        if np.issubdtype(community_cov.dtype, np.number):
            if not np.any(np.isnan(community_cov)) and not np.any(pd.isnull(community_cov)):
                if empirical:
                    cov = emp_cov
                    mu = np.zeros_like(df_mu.values)
                else:
                    mu = df_mu.values  # <--- Hack
                    cov = 0.9*community_cov

                # TODO: Use mu to alter cov

                truth = df_truth.loc[expiry].values
                w = port(cov=cov)
                cov_met = cov_metric(mu=mu, cov=cov,truth=truth, **cov_metric_kwargs)
                try:
                    cov_mets.append(np.log(1e-8 + cov_met))
                except ArithmeticError:
                    raise Exception('log is a problem')

                port_ret = np.log(np.dot(w, np.exp(truth)))
                if np.isnan(port_ret):
                    raise ValueError('port return is nan')
                port_rets.append(port_ret)

    port_score = port_metric(port_rets)
    sum_cov_met = np.sum(cov_mets)

    return port_score, sum_cov_met


def metric_leaderboard(portfolio_methods, port_metric, cov_metric, cov_metric_kwargs, empirical=False):
    port_metrics = {}
    cov_metrics = {}
    for port in portfolio_methods:
        print(f"Running backtest for {port.__name__}")
        port_met, cov_met = precise_backtest(port=port, port_metric=port_metric, cov_metric=cov_metric, empirical=empirical, cov_metric_kwargs=cov_metric_kwargs)
        port_metrics[port.__name__] = port_met
        cov_metrics[port.__name__] = cov_met

    # Sort the results by total return from highest to lowest
    sorted_port_results = dict(sorted(port_metrics.items(), key=lambda item: item[1], reverse=True))
    sorted_cov_results = dict(sorted(cov_metrics.items(), key=lambda item: item[1], reverse=True))

    return sorted_port_results, sorted_cov_results


def community_covariance_comparison(portfolio_methods, port_metric, cov_metric, cov_metric_kwargs):
    print(f"Comparing metric: {port_metric.__name__}")

    emp_port_results, emp_cov_results = metric_leaderboard(portfolio_methods, port_metric=port_metric, cov_metric=cov_metric, empirical=True, cov_metric_kwargs=cov_metric_kwargs)
    community_port_results, community_cov_results = metric_leaderboard(portfolio_methods, port_metric=port_metric, cov_metric=cov_metric, empirical=False, cov_metric_kwargs=cov_metric_kwargs)

    print({'empirical':emp_cov_results})
    print({'community':community_cov_results})

    # Create a DataFrame to align port results for comparison
    all_port_methods = list(set(emp_port_results.keys()).union(set(community_port_results.keys())))
    port_comparison_df = pd.DataFrame(index=all_port_methods, columns=['Empirical cov', 'Community cov'])
    for method in all_port_methods:
        port_comparison_df.loc[method, 'Empirical cov'] = emp_port_results.get(method, 'N/A')
        port_comparison_df.loc[method, 'Community cov'] = community_port_results.get(method, 'N/A')
    port_comparison_df['Community helps'] = port_comparison_df.apply(lambda row: 1 if row['Community cov'] > row['Empirical cov'] else 0,
                                                  axis=1)



    print(f"Comparison of {port_metric.__name__} using Empirical vs. Community Covariance:")
    print(port_comparison_df)
    return port_comparison_df


def show_leaderboard():
    metric= sharpe_ratio
    leaderboard = metric_leaderboard(portfolio_methods, empirical=True, port_metric= metric)
    print("Leaderboard: "+metric.__name__)
    for method, total_ret in leaderboard.items():
        print(f"{method}: {total_ret:.4f}")


if __name__=='__main__':
    dim = 3
    port_metric = sharpe_ratio
    cov_metric = projected_subspace_likelihood
    cov_metric_kwargs ={'dim':dim, 'num_subspaces':100}
    community_covariance_comparison(portfolio_methods, port_metric=port_metric, cov_metric=cov_metric, cov_metric_kwargs=cov_metric_kwargs)


