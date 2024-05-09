import sys

path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)
import ising

import numpy as np


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
