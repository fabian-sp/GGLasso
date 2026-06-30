import numpy as np
from scipy.special import comb

def subsample(X, N):
    """Generate N subsamples from the given data matrix X.

    This function takes a data matrix X with p variables (rows) and
    n observations (columns) and returns N subsamples, each containing
    a subset of the observations.

    Parameters:
      - X (np.ndarray): Data matrix of shape (p, n) where p is the number
        of variables and n is the number of observations.
      - N (int): Number of desired subsamples.

    Returns:
      - np.ndarray: An array containing subsampled data of shape (p, b, N),
        where b is determined based on the number of observations n.

    Raises:
      - ValueError: If the number of observations n is less than N.
    """

    p, n = X.shape

    if n < N:
        raise Exception("The number of samples n must be at least N")

    ### SpiecEasi (huge) implementation of StARS
    ### https://github.com/zdk123/SpiecEasi/blob/master/R/select.R
    sample_ratio = 0.8 if n <= 144 else 10 * np.sqrt(n) / n

    num_samples = int(np.floor(n * sample_ratio))

    ### Create an array for subsamples with shape (p, b(n), N)
    subsamples = np.empty((p, num_samples, N))

    ### Subsample without replacement
    for i in range(N):
        samples_ixs = np.random.choice(n, size=num_samples, replace=False)
        subsamples[:, :, i] = X[:, samples_ixs]

    return subsamples


def estimate_instability(subsamples, estimator, lmbda, return_estimates=False):
    """Estimate the instability using a set of subsamples, as in
    (https://arxiv.org/pdf/1006.3316.pdf, page 6)

    Parameters:
      - subsamples (np.array): the subsample array
      - estimator (function): the estimator to be used. See
        documentation for stars.fit for more info.
      - lmbda (float): the regularization parameter at which to run
        the estimator
      - return_estimates (bool, optional): If the estimate for each
        subsample should also be returned. Defaults to False.

    Returns:
      - instability (float): The estimated total instability

    """
    estimates = estimator(subsamples, lmbda)
    p = subsamples.shape[0]
    edge_average = np.mean(estimates, axis=0)

    ### the variance of a Bernoulli distribution
    edge_instability = 2 * edge_average * (1-edge_average)

    # In the following, the division by 2 is to account for counting
    # every edge twice (as the estimate matrix is symmetric)
    total_instability = np.sum(edge_instability, axis=(0,1)) / comb(p, 2) / 2
    if return_estimates:
        return total_instability, estimates
    else:
        return total_instability