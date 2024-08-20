from scipy.stats.qmc import MultivariateNormalQMC
from sklearn.covariance import EmpiricalCovariance
import numpy as np
from datetime import datetime, timedelta
from monteprediction import SPDR_ETFS
import yfinance as yf
from functools import lru_cache


def wednesday_two_back(expiry):
    expiry_date = datetime.strptime(expiry, '%Y_%m_%d')
    offset = (expiry_date.weekday() - 2) % 7
    prior_wednesday = expiry_date - timedelta(days=offset+7)
    return prior_wednesday.date()


@lru_cache(maxsize=None)
def yahoo_empirical_cov(expiry):
    wed_two_back = wednesday_two_back(expiry=expiry)
    num_weeks = int(52 + 2 * 52 * np.random.rand())
    start_date = wed_two_back - timedelta(weeks=num_weeks)
    data = yf.download(SPDR_ETFS, start=start_date, end=wed_two_back, interval="1wk")
    weekly_prices = data['Adj Close']
    weekly_returns = weekly_prices.pct_change().dropna()

    # Use cov estimation to generate samples
    cov_matrix = EmpiricalCovariance().fit(weekly_returns).covariance_
    return cov_matrix
