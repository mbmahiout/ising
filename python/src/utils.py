import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
import os
from IPython.display import display


def get_all_recording_means(all_samples: list) -> list:
    return [
        [sample.getMeans() for sample in all_samples[i]]
        for i in range(len(all_samples))
    ]


def get_all_recording_pcorrs(all_samples: list) -> list:
    return [
        [sample.getPairwiseCorrs().flatten() for sample in all_samples[i]]
        for i in range(len(all_samples))
    ]


def get_rmse(true: np.array, predicted: np.array) -> float:
    return np.sqrt(np.mean((true - predicted) ** 2))


def get_r_squared(true: np.array, predicted: np.array) -> float:
    sst = np.sum((true - np.mean(true)) ** 2)
    ssr = np.sum((true - predicted) ** 2)
    return 1 - (ssr / sst)


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


########################################################################################


def get_ks_sensitivity_results(samples: list, labels: list):
    states = [s.getStates() for s in samples]
    num_recs = len(labels)
    test_stats = np.zeros((num_recs, num_recs))
    pvals = np.zeros((num_recs, num_recs))

    for i in range(num_recs):
        for j in range(num_recs):
            stat, pval = ks_2samp(states[i].flatten(), states[j].flatten())
            if isinstance(stat, np.generic):
                stat = stat.item()
            if isinstance(pval, np.generic):
                pval = pval.item()
            test_stats[i, j] = stat
            pvals[i, j] = pval

    test_stats = pd.DataFrame(test_stats, index=labels, columns=labels)
    pvals = pd.DataFrame(pvals, index=labels, columns=labels)

    pd.options.display.float_format = "{:.2e}".format

    return test_stats, pvals


def make_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' was created.")
        else:
            print(f"Directory '{dir_path}' already exists.")
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")


def do_ks_sensitivity_tests(
    labels,
    all_samples,
    sample_names,
    mouse_name,
    sens_param,
    path,
):

    print(f"{mouse_name}, {sens_param} sensitivity")

    for (
        samples,
        sample_name,
    ) in zip(all_samples, sample_names):
        test_stats, pvals = get_ks_sensitivity_results(samples, labels)

        print(sample_name + ":")

        print("test statistics:")
        display(test_stats)

        print("p-values:")
        display(pvals)

        test_stats.to_csv(path + f"{sample_name}_tstats.csv")
        pvals.to_csv(path + f"{sample_name}_pvals.csv")
