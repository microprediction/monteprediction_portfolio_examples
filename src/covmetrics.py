from scipy.stats import multivariate_normal
import numpy as np
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from scipy.special import ive  # Modified Bessel function


def likelihood(mu, cov, truth):
    the_mean = np.array(mu).flatten()
    rv = multivariate_normal(mean=the_mean, cov=cov)
    return rv.pdf(truth)


def cov_likelihood(mu, cov, truth):
    rv = multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
    return rv.pdf(truth)


def subspace_likelihood(mu, cov, truth, num_subspaces=100, dim=3):
    return subspace_likelihood_factory(mu=mu, cov=cov, truth=truth, num_subspaces=num_subspaces,dim=dim, project=False)


def projected_subspace_likelihood(mu, cov, truth, num_subspaces=100, dim=3):
    return subspace_likelihood_factory(mu=mu, cov=cov, truth=truth, num_subspaces=num_subspaces, dim=dim, project=True)


def subspace_likelihood_factory(mu, cov, truth, num_subspaces, project, dim):
    the_mean = np.array(mu).flatten()
    n = cov.shape[0]
    total_log_likelihood = 0

    for _ in range(num_subspaces):
        indices = np.random.choice(n, size=dim, replace=False)
        sub_cov = cov[np.ix_(indices, indices)]
        sub_mu = the_mean[indices]
        sub_truth = truth[indices]

        # Compute likelihood for the subspace
        if project:
            log_likelihood = np.log(projected_likelihood(sub_mu, sub_cov, sub_truth))
        else:
            log_likelihood = np.log(likelihood(sub_mu, sub_cov, sub_truth))
        total_log_likelihood += log_likelihood

    # Average likelihood
    avg_log_like = total_log_likelihood / num_subspaces
    return np.exp(avg_log_like)


def projected_likelihood(mu, cov, truth):
    """
    Compute the approximate density of y under the projected normal distribution.

    Parameters:
        truth (ndarray): Unit vector on the n-sphere.
        mu (ndarray): Mean direction vector (can be zero).
        cov (ndarray): Covariance matrix.

    Returns:
        float: Density value, accounting for scale invariance.
    """
    projected_cov = truth.T @ np.linalg.inv(cov) @ truth
    density = np.exp(-0.5 * projected_cov) / np.sqrt(np.linalg.det(cov))
    return density


COV_METRICS = [projected_likelihood, likelihood, subspace_likelihood, projected_subspace_likelihood, cov_likelihood]
