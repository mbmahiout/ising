import sys

path2cpp_pkg = (
    "/Users/mariusmahiout/Documents/repos/ising_core/build"  # to-do: make relative
)
sys.path.append(path2cpp_pkg)
import ising

from utils import get_rmse, int_linspace
from misc_plotting import plot_generalized

import numpy as np
import pandas as pd
from tqdm import tqdm


# indep. pair couplings
def get_indep_pairs(num_units: int) -> np.ndarray:
    if num_units % 2 == 0:
        pairs = np.split(np.random.permutation(num_units), num_units / 2)
    else:
        pairs = np.split(np.random.permutation(num_units - 1), (num_units - 1) / 2)
    return pairs


def get_ip_couplings(num_units: int, pairs: np.ndarray, beta: float) -> np.ndarray:
    J = np.zeros((num_units, num_units))
    for pair in pairs:
        J[pair[0]][pair[1]] = np.random.normal(0, beta / np.sqrt(num_units))
        J[pair[1]][pair[0]] = J[pair[0]][pair[1]]
    return J


def get_ip_model(num_units: int, pairs: np.ndarray, beta: float) -> ising.EqModel:
    h = np.random.uniform(-0.3 * beta, 0.3 * beta, num_units)
    J = get_ip_couplings(num_units, pairs, beta)
    return ising.EqModel(J, h)


def get_empirical_ip_obs(
    model: ising.EqModel,
    num_sims: int,
    num_burn: int = 1000,
) -> dict:
    sims = model.simulate(num_sims, num_burn)
    return {"m": sims.getMeans(), "chi": sims.getPairwiseCorrs()}


# analytic observables
def get_pair_hamiltonian(s_i, s_j, J_ij, h_i, h_j):
    return -J_ij * s_i * s_j - h_i * s_i - h_j * s_j


def get_pair_partition_func(h, J, i, j):
    Z = (
        np.exp(-get_pair_hamiltonian(+1, +1, J[i, j], h[i], h[j]))
        + np.exp(-get_pair_hamiltonian(+1, -1, J[i, j], h[i], h[j]))
        + np.exp(-get_pair_hamiltonian(-1, +1, J[i, j], h[i], h[j]))
        + np.exp(-get_pair_hamiltonian(-1, -1, J[i, j], h[i], h[j]))
    )
    return Z


def get_m_pair(h, J, i, j):
    num = (
        np.exp(-get_pair_hamiltonian(+1, +1, J[i, j], h[i], h[j]))
        + np.exp(-get_pair_hamiltonian(+1, -1, J[i, j], h[i], h[j]))
        - np.exp(-get_pair_hamiltonian(-1, +1, J[i, j], h[i], h[j]))
        - np.exp(-get_pair_hamiltonian(-1, -1, J[i, j], h[i], h[j]))
    )
    m_ij = num / get_pair_partition_func(h, J, i, j)
    return m_ij


def get_chi_pair(h, J, i, j):
    num = (
        np.exp(-get_pair_hamiltonian(+1, +1, J[i, j], h[i], h[j]))
        - np.exp(-get_pair_hamiltonian(+1, -1, J[i, j], h[i], h[j]))
        - np.exp(-get_pair_hamiltonian(-1, +1, J[i, j], h[i], h[j]))
        + np.exp(-get_pair_hamiltonian(-1, -1, J[i, j], h[i], h[j]))
    )
    chi_ij = num / get_pair_partition_func(h, J, i, j)
    return chi_ij


def get_analytic_means(h: np.ndarray, J: np.ndarray, pairs: list) -> np.ndarray:
    num_units = h.shape[0]
    m = np.zeros(num_units)

    for i, j in pairs:
        m[i] = get_m_pair(h, J, i, j)
        m[j] = get_m_pair(h, J, j, i)

    return m


def get_analytic_pcorrs(h: np.ndarray, J: np.ndarray, pairs: list) -> np.ndarray:
    num_units = h.shape[0]
    chi = np.ones((num_units, num_units))

    m = get_analytic_means(h, J, pairs)
    for i in range(num_units):
        for j in [k for k in range(num_units) if k != i]:
            chi[i, j] = m[i] * m[j]

    for i, j in pairs:
        chi[i, j] = get_chi_pair(h, J, i, j)
        chi[j, i] = chi[i, j]

    return chi


def get_analytic_ip_obs(h: np.ndarray, J: np.ndarray, pairs: np.ndarray) -> dict:
    return {
        "m": get_analytic_means(h, J, pairs),
        "chi": get_analytic_pcorrs(h, J, pairs),
    }


# sensitivity analysis


def get_emp_vs_analytic_rmses(
    num_units: int, beta: float, num_sims: int, num_burn: int
):
    pairs = get_indep_pairs(num_units)
    model = get_ip_model(num_units, pairs, beta)
    emp_obs = get_empirical_ip_obs(model, num_sims, num_burn)
    anal_obs = get_analytic_ip_obs(model.getFields(), model.getCouplings(), pairs)

    rmse_m = get_rmse(anal_obs["m"], emp_obs["m"])
    rmse_chi = get_rmse(anal_obs["chi"], emp_obs["chi"])

    return rmse_m, rmse_chi


def get_sampler_rmses_num_units(
    min_units: int,
    max_units: int,
    beta: float,
    num_sims: int,
    num_burn: int,
    num_runs: int,
):
    rmses_means = dict((run, []) for run in range(num_runs))
    rmses_pcorrs = dict((run, []) for run in range(num_runs))
    nunits_range = range(min_units, max_units + 1)
    for run in tqdm(range(num_runs)):
        for num_units in nunits_range:
            rmse_m, rmse_chi = get_emp_vs_analytic_rmses(
                num_units, beta, num_sims, num_burn
            )

            rmses_means[run].append(rmse_m)
            rmses_pcorrs[run].append(rmse_chi)

    rmses_means = pd.DataFrame(rmses_means)
    rmses_pcorrs = pd.DataFrame(rmses_pcorrs)
    return {
        "RMSEs means": rmses_means,
        "RMSEs pcorrs": rmses_pcorrs,
        "nunits range": list(nunits_range),
    }


def get_sampler_rmses_beta(
    beta_min: float,
    beta_max: float,
    num_units: int,
    num_sims: int,
    num_burn: int,
    num_runs: int,
):
    rmses_means = dict((run, []) for run in range(num_runs))
    rmses_pcorrs = dict((run, []) for run in range(num_runs))
    betas_range = np.linspace(beta_min, beta_max, 100)

    for run in tqdm(range(num_runs)):
        for beta in betas_range:
            rmse_m, rmse_chi = get_emp_vs_analytic_rmses(
                num_units, beta, num_sims, num_burn
            )

            rmses_means[run].append(rmse_m)
            rmses_pcorrs[run].append(rmse_chi)

    rmses_means = pd.DataFrame(rmses_means)
    rmses_pcorrs = pd.DataFrame(rmses_pcorrs)
    return {
        "RMSEs means": rmses_means,
        "RMSEs pcorrs": rmses_pcorrs,
        "betas range": betas_range,
    }


def get_sampler_rmses_num_sims(
    sims_min: int,
    sims_max: int,
    num_units: int,
    beta: float,
    num_burn: int,
    num_runs: int,
):
    rmses_means = dict((run, []) for run in range(num_runs))
    rmses_pcorrs = dict((run, []) for run in range(num_runs))
    nsims_range = int_linspace(sims_min, sims_max)

    for run in tqdm(range(num_runs)):
        for num_sims in nsims_range:
            rmse_m, rmse_chi = get_emp_vs_analytic_rmses(
                num_units, beta, num_sims, num_burn
            )

            rmses_means[run].append(rmse_m)
            rmses_pcorrs[run].append(rmse_chi)

    rmses_means = pd.DataFrame(rmses_means)
    rmses_pcorrs = pd.DataFrame(rmses_pcorrs)
    return {
        "RMSEs means": rmses_means,
        "RMSEs pcorrs": rmses_pcorrs,
        "nsims range": nsims_range,
    }


def run_sensitivity_analysis(
    path="./analyses/sampler_sensitivity/test.pdf",
    num_runs: int = 10,
    min_units: int = 3,
    max_units: int = 200,
    beta: float = 1.3,
    num_sims: int = 15_000,
    num_burn: int = 1000,
    beta_min: float = 0.1,
    beta_max: float = 10,
    num_units: int = 20,
    sims_min: int = 1000,
    sims_max: int = 100_000,
):
    # iterate over num_units
    out_nunits = get_sampler_rmses_num_units(
        min_units, max_units, beta, num_sims, num_burn, num_runs
    )

    # iterate over beta
    out_betas = get_sampler_rmses_beta(
        beta_min, beta_max, num_units, num_sims, num_burn, num_runs
    )

    # iterate over num_sims
    out_nsims = get_sampler_rmses_num_sims(
        sims_min, sims_max, num_units, beta, num_burn, num_runs
    )

    # plotting
    layout_spec = {
        (1, 1): {
            "data": out_nunits["RMSEs means"].to_numpy(),
            "label": r"$\Large \text{RMSE}(m_i)$",
            "steps": out_nunits["nunits range"],
            "step_label": r"",
        },
        (2, 1): {
            "data": out_nunits["RMSEs pcorrs"].to_numpy(),
            "label": r"$\Large \text{RMSE}(\chi_{ij})$",
            "steps": out_nunits["nunits range"],
            "step_label": r"$\Large N$",
        },
        (1, 2): {
            "data": out_betas["RMSEs means"].to_numpy(),
            "label": r"",
            "steps": out_betas["betas range"],
            "step_label": r"",
        },
        (2, 2): {
            "data": out_betas["RMSEs pcorrs"].to_numpy(),
            "label": r"",
            "steps": out_betas["betas range"],
            "step_label": r"$\Large \beta$",
        },
        (1, 3): {
            "data": out_nsims["RMSEs means"].to_numpy(),
            "label": r"",
            "steps": out_nsims["nsims range"],
            "step_label": r"",
        },
        (2, 3): {
            "data": out_nsims["RMSEs pcorrs"].to_numpy(),
            "label": r"",
            "steps": out_nsims["nsims range"],
            "step_label": r"$\Large M$",
        },
    }

    plot_generalized(layout_spec, path=path)
