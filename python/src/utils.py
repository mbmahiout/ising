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


def stable_arctanh(x, epsilon=1e-10):
    x_clipped = np.clip(x, -1 + epsilon, 1 - epsilon)
    return np.arctanh(x_clipped)


def stable_log(x, epsilon=1e-10):
    nan_mask = np.isnan(x)
    if np.all(nan_mask):
        raise ValueError("All values are NaN")
    else:
        mean_val = np.nanmean(x)
        x_stabilized = np.where(nan_mask, mean_val, x)

    max_non_inf = np.max(x_stabilized[np.isfinite(x_stabilized)])
    x_clipped = np.clip(x_stabilized, epsilon, max_non_inf)
    return np.log(x_clipped)


def get_inv_mat(mat, size):
    try:
        inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:  # singular matrix
        I = np.identity(size)
        inv_mat = np.linalg.lstsq(mat, I, rcond=None)[0]
    return inv_mat


def get_analysis_path(analysis_name, num_units, bin_width):
    analysis_path = "./analyses/"
    dir_name = f"n{num_units}b{bin_width}{analysis_name}"
    analysis_path += f"./{dir_name}/"
    return analysis_path


def int_linspace(min_int: int, max_int: int, num_ints: int = 100):
    linspace_floats = np.linspace(min_int, max_int, num_ints)
    return np.round(linspace_floats).astype(int)


def get_metadata(
    num_units, is_empirical_analysis, eq_inv_methods=[], neq_inv_methods=[], **kwargs
):
    metadata = {}
    metadata["num_units"] = num_units

    if is_empirical_analysis:
        bin_width = kwargs["bin_width"]
        num_bins = kwargs["num_bins"]
        metadata["bin_width"] = bin_width
        metadata["num_bins"] = num_bins
    else:
        num_sims = kwargs["num_sims"]
        num_burn = kwargs.get("num_burn", 1000)
        true_fields = kwargs["true_fields"]
        true_couplings = kwargs["true_couplings"]
        metadata["true_model"] = {
            "true_fields": true_fields,
            "true_couplings": true_couplings,
            "num_sims": num_sims,
            "num_burn": num_burn,
        }

    if (eq_inv_methods != []) or (neq_inv_methods != []):
        metadata["inverse_methods"] = {
            "EQ": eq_inv_methods,
            "NEQ": neq_inv_methods,
        }
        if ("ML" in eq_inv_methods) or ("ML" in neq_inv_methods):
            # each can be dict if multiple ML models with different hyperparams
            num_steps = kwargs["num_steps"]
            learning_rate = kwargs["learning_rate"]
            is_converged = kwargs["is_converged"]
            metadata["maximum_likelihood"] = {
                "num_steps": num_steps,
                "learning_rate": learning_rate,
                "is_converged": is_converged,
            }
        if "ML" in eq_inv_methods:
            num_sims_ml = kwargs["num_sims_ml"]
            num_burn_ml = kwargs.get("num_burn_ml", 1000)
            metadata["maximum_likelihood"]["num_sims"] = num_sims_ml
            metadata["maximum_likelihood"]["num_burn"] = num_burn_ml
    return metadata
