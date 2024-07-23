import sys

path2cpp_pkg = (
    "/Users/mariusmahiout/Documents/repos/ising_core/build"  # change to relative import
)
sys.path.append(path2cpp_pkg)
import ising

import numpy as np
from scipy.io import loadmat

########
# core #
########


def get_recording_sample(
    fname,
    mouse_name,
    bin_width=50,
    num_units=None,
    is_shuffled_units=False,
    is_shuffled_bins=False,
    data_dir="data",
):
    mat_dict = load_recordings(fname, mouse_name, data_dir)
    states = get_recording_states(mat_dict)
    states = reduce_time_resolution(states, bin_width)

    if is_shuffled_units:
        states = shuffle_units(states)

    if is_shuffled_bins:
        states = shuffle_bins(states)

    if num_units is not None:
        states = subset_units(states, num_units)

    states = binary2ising(states)
    return ising.Sample(states.T)


def get_partitioned_sample(
    fname,
    mouse_name,
    bin_width=50,
    num_subsamples=1,
    data_dir="data",
):
    mat_dict = load_recordings(fname, mouse_name, data_dir)
    states = get_recording_states(mat_dict)
    states = reduce_time_resolution(states, bin_width)
    subsamples = get_nonoverlapping_subsamples(states, num_subsamples)
    subsamples = [binary2ising(s) for s in subsamples]
    subsamples = [ising.Sample(s) for s in subsamples]
    return subsamples  # list(map(lambda s: ising.Sample(binary2ising(s)), subsamples))


#############
# auxiliary #
#############


def load_recordings(fname, mouse_name, data_dir="data"):
    path = f"./{data_dir}/{mouse_name}/{fname}"
    mat_dict = loadmat(path)
    return mat_dict


def get_recording_states(mat_dict):
    states = []
    neuron_keys = list(filter(lambda key: "firingratedata" in key, mat_dict.keys()))
    for key in neuron_keys:
        states.append(mat_dict[key][0])
    return np.array(states).T


def get_nonoverlapping_subsamples(states, num_subsamples):
    num_units = states.shape[1]
    num_units_subsample = num_units // num_subsamples

    subsamples = []
    start_idx = 0

    for _ in range(num_subsamples):
        end_idx = start_idx + num_units_subsample
        subsamples.append(states[:, start_idx:end_idx])
        start_idx = end_idx

    return subsamples


def reduce_time_resolution(states, bin_width, bin_width_orig=50):
    if bin_width == bin_width_orig:
        return states
    elif bin_width > bin_width_orig:
        num_bins_orig = states.shape[0]
        bin_width_ratio = bin_width / bin_width_orig
        num_bins = int(np.floor(num_bins_orig / bin_width_ratio))

        num_units = states.shape[1]
        lo_res_states = np.zeros((num_bins, num_units))
        for bin in range(num_bins):
            start_idx = int(bin * bin_width_ratio)
            end_idx = int((bin + 1) * bin_width_ratio)
            lo_res_states[bin, :] = np.sum(states[start_idx:end_idx, :], axis=0)
        lo_res_states = np.heaviside(lo_res_states - 0.5, 0)
        return lo_res_states
    else:
        raise ValueError(f"bin_width must be >= {bin_width_orig} ms")


def binary2ising(states):
    return 2 * states - 1


def ising2binary(states):
    return 0.5 * (states + 1)


def subset_units(states, num_units):
    num_units_orig = states.shape[1]
    if num_units > num_units_orig:
        raise ValueError(f"num_units must be < {num_units_orig}")
    return states[:, :num_units]


def shuffle_bins(states):
    num_bins = states.shape[0]
    indices = np.random.permutation(num_bins)
    shuffled_states = states[indices, :]
    return shuffled_states


def shuffle_units(states):
    num_units = states.shape[1]
    indices = np.random.permutation(num_units)
    shuffled_states = states[:, indices]
    return shuffled_states
