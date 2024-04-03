import numpy as np


def get_exp_terms(h: np.ndarray, J: np.ndarray, i: int, j: int):
    term1 = np.exp(h[i] + h[j] + J[i, j])
    term2 = np.exp(h[i] - h[j] - J[i, j])
    term3 = np.exp(-h[i] + h[j] - J[i, j])
    term4 = np.exp(-h[i] - h[j] + J[i, j])
    return term1, term2, term3, term4


def get_analytic_corrs(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    num_units = h.shape[0]
    chi = np.ones((num_units, num_units))
    for i in range(num_units):
        for j in [k for k in range(num_units) if k != i]:
            trm1, trm2, trm3, trm4 = get_exp_terms(h, J, i, j)
            chi[i, j] = (trm1 - trm2 - trm3 + trm4) / (trm1 + trm2 + trm3 + trm4)
    return chi


def get_rmse(true, predicted):
    return np.sqrt(np.mean((true - predicted) ** 2))


# to compare with Nguyen et al.
def get_reconstruction_err(true_couplings, est_couplings):
    num_units = true_couplings.shape[0]

    true_sq_sum = 0
    diff_sq_sum = 0
    for i in range(num_units):
        for j in range(i + 1, num_units):
            true_sq_sum += true_couplings[i, j] ** 2
            diff_sq_sum += (est_couplings[i, j] - true_couplings[i, j]) ** 2
    return np.sqrt(diff_sq_sum / true_sq_sum)