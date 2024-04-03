import numpy as np
import matplotlib.pyplot as plt


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
    max_val = max(max(gt), max(est))
    min_val = min(min(gt), min(est))
    interval = np.linspace(min_val*2, max_val*2, 100)

    plt.scatter(gt, est, color="dodgerblue", alpha=.75)
    plt.plot(interval, interval, linestyle="--", color="black")

    plt.xlim([min_val*1.2, max_val*1.2])
    plt.ylim([min_val*1.2, max_val*1.2])
    plt.show()
    plt.close()