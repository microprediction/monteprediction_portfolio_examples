import numpy as np


# Portfolio metrics

def total_return(weekly_log_returns):
    return np.exp(np.sum(weekly_log_returns)) - 1


def std_return(weekly_log_returns):
    return np.std(weekly_log_returns)


def mean_return(weekly_log_returns):
    return np.mean(weekly_log_returns)


def sharpe_ratio(weekly_log_returns, risk_free_rate=0):
    # Assuming weekly data; annualize by multiplying by sqrt(52)
    mean_return = np.mean(weekly_log_returns)
    std_dev = np.std(weekly_log_returns)
    return (mean_return - risk_free_rate) / std_dev * np.sqrt(52)


def sortino_ratio(weekly_log_returns, risk_free_rate=0):
    # Assuming weekly data; annualize by multiplying by sqrt(52)
    mean_return = np.mean(weekly_log_returns)
    downside_returns = [r for r in weekly_log_returns if r < mean_return]
    downside_deviation = np.std(downside_returns)
    return (mean_return - risk_free_rate) / downside_deviation * np.sqrt(52)


PORT_METRICS = [total_return, sortino_ratio, sharpe_ratio, mean_return, std_return]
