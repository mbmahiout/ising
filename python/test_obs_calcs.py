import numpy as np
import time
import matplotlib.pyplot as plt

import sys
path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)
import ising


def scatter_compare_2d_obs(gt: np.ndarray, est: np.ndarray):
    num_units = gt.shape[0]
    gt2plt = []
    est2plt = []
    for i in range(num_units):
        for j in [k for k in range(num_units) if k != i]:
            gt2plt.append(gt[i, j])
            est2plt.append(est[i, j])

    max_val = max(max(gt2plt), max(est2plt))
    min_val = min(min(gt2plt), min(est2plt))
    interval = np.linspace(min_val*2, max_val*2, 100)

    plt.scatter(gt2plt, est2plt, color="dodgerblue", alpha=.75)
    plt.plot(interval, interval, linestyle="--", color="black")

    plt.xlim([min_val*1.2, max_val*1.2])
    plt.ylim([min_val*1.2, max_val*1.2])
    plt.show()
    plt.close()


def scatter_compare_1d_obs(gt: np.ndarray, est: np.ndarray):
    num_units = gt.shape[0]

    max_val = max(max(gt), max(est))
    min_val = min(min(gt), min(est))
    interval = np.linspace(min_val*2, max_val*2, 100)

    plt.scatter(gt, est, color="dodgerblue", alpha=.75)
    plt.plot(interval, interval, linestyle="--", color="black")

    plt.xlim([min_val*1.2, max_val*1.2])
    plt.ylim([min_val*1.2, max_val*1.2])
    plt.show()
    plt.close()


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


def get_pairwise_corrs(sample):
    num_bins = sample.shape[1]
    return np.matmul(sample, sample.T) / num_bins


def get_means(sample):
    return np.mean(sample, axis=1)


# setting up model
num_units = 100
beta = 1.3
h = np.random.uniform(-.3 * beta, .3 * beta, num_units)
J = np.zeros((num_units, num_units))
if num_units % 2 == 0:
    pairs = np.split(np.random.permutation(num_units), num_units / 2)
else:
    pairs = np.split(np.random.permutation(num_units - 1), (num_units - 1) / 2)
for pair in pairs:
    J[pair[0]][pair[1]] = np.random.normal(0,  beta / np.sqrt(num_units))
    J[pair[1]][pair[0]] = J[pair[0]][pair[1]]

eq_model = ising.EqModel(num_units, J, h)

# simulating
t0 = time.time()

sim = eq_model.simulate(150000, 10000)

t1 = time.time()
dt = t1 - t0
print("Simulation took {:.2f} seconds.".format(dt))


# computing observables
# chi_gt = get_analytic_corrs(h, J)

t0 = time.time()

m_est_py = get_means(sim)

t1 = time.time()
dt = t1 - t0
print("Computing observables (Python) took {:.2f} seconds.".format(dt))

t0 = time.time()

m_est_cpp = ising.getMeans(sim)

t1 = time.time()
dt = t1 - t0
print("Computing observables (C++) took {:.2f} seconds.".format(dt))

# chi_gt = get_analytic_corrs(h, J)

# t0 = time.time()

# chi_est_py = get_pairwise_corrs(sim)

# t1 = time.time()
# dt = t1 - t0
# print("Computing observables (Python) took {:.2f} seconds.".format(dt))

# t0 = time.time()

# chi_est_cpp = ising.getPairwiseCorrs(sim)

# t1 = time.time()
# dt = t1 - t0
# print("Computing observables (C++) took {:.2f} seconds.".format(dt))



# plotting
print(m_est_cpp.shape, m_est_py.shape)
scatter_compare_1d_obs(m_est_cpp, m_est_py)

