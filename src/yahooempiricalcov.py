from scipy.stats.qmc import MultivariateNormalQMC
from sklearn.covariance import EmpiricalCovariance
import numpy as np
from datetime import datetime, timedelta
from monteprediction import SPDR_ETFS
import yfinance as yf
from functools import lru_cache
import time

YEARS = 2

def wednesday_two_back(expiry):
    expiry_date = datetime.strptime(expiry, '%Y_%m_%d')
    offset = (expiry_date.weekday() - 2) % 7
    prior_wednesday = expiry_date - timedelta(days=offset+7)
    return prior_wednesday.date()




@lru_cache(maxsize=None)
def yahoo_empirical_cov(expiry):
    retries = 5
    delay = 2
    backoff = 2
    last_exception = None

    for attempt in range(retries):
        try:
            wed_two_back = wednesday_two_back(expiry=expiry)
            num_weeks = int(YEARS * 52)
            start_date = wed_two_back - timedelta(weeks=num_weeks)
            data = yf.download(SPDR_ETFS, start=start_date, end=wed_two_back, interval="1wk")
            weekly_prices = data['Adj Close']
            weekly_returns = weekly_prices.pct_change().dropna()

            # Use cov estimation to generate samples
            cov_matrix1 = EmpiricalCovariance().fit(weekly_returns).covariance_
            from covestimation import cov_estimation
            cov_matrix = cov_estimation(weekly_returns)
            return cov_matrix

        except Exception as e:
            last_exception = e
            print(f"Error occurred: {e}. Attempt {attempt + 1} of {retries}.")
            time.sleep(delay)
            delay *= backoff  # Increase delay with each retry

    # Raise the last exception if all retries fail
    raise last_exception
