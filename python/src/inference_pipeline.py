import sys
import os

path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)

import ising
import os
import sys
import time

import preprocessing as pre
import model_eval as eval
import utils as utils
import misc_plotting as misc_plotting
import isingfitter as fitter

import numpy as np
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--is_testing",
    type=bool,
    help="Just running a test? (won't do as many inference steps)",
    default=True,
)
parser.add_argument(
    "--mouse_name", type=str, help="Name of the recorded mouse.", default="Angie"
)
parser.add_argument("--num_units", type=int, help="Number of units.", default=10)
parser.add_argument("--num_sims", type=int, help="Name of simulations.", default=15_000)
args = parser.parse_args()

# variable hyperparams
is_testing = args.is_testing
mouse_name = args.mouse_name
num_units = args.num_units
num_sims = args.num_sims

# fixed hyperparams
num_burn = 5_000
lr = 0.1
win_size = 10
tol_ml = 1e-3
tol_pl = 1e-9

bin_width = 50  # ms
sample = pre.get_recording_sample(  # should be able to choose which!
    fname="RESULTS_Angie_20170825_1220_allbeh_1000s.mat",
    mouse_name=mouse_name,
    bin_width=bin_width,
    num_units=num_units,
)

path = "analyses"
utils.make_dir(path)

###############
# EQUILIBRIUM #
###############
print("\n")

# setting up model
h_init = np.random.uniform(-1.5, 1.5, num_units)
J_init = np.random.normal(0, 1, (num_units, num_units))
J_init = (J_init.T + J_init) * np.sqrt(2) / 2
np.fill_diagonal(J_init, 0)

eq_model = ising.EqModel(J_init, h_init)


# inference
eq_fitter = fitter.EqFitter(eq_model)

print("Starting inference (EQ)")
print("TAP")
t1 = time.time()
eq_fitter.TAP(sample)
t2 = time.time()
utils.print_elapsed_time(t1, t2)

print("Likelihood maximization")
t1 = time.time()
eq_fitter.maximize_likelihood(
    sample=sample,
    max_steps=4_000,
    learning_rate=0.005,
    win_size=win_size,
    tolerance=tol_ml,
    num_sims=num_sims,
    num_burn=num_burn,
    calc_llh=False,
)
print("Run 1 - eta = 0.005, DONE")
t2 = time.time()
utils.print_elapsed_time(t1, t2)

if not is_testing:
    t1 = time.time()
    eq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=4_000,
        learning_rate=0.0025,
        win_size=win_size,
        tolerance=tol_ml,
        num_sims=num_sims,
        num_burn=num_burn,
        calc_llh=False,
    )
    print("Run 2 - eta = 0.0025, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

    t1 = time.time()
    eq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=10_000,
        learning_rate=0.001,
        win_size=win_size,
        tolerance=tol_ml,
        num_sims=num_sims,
        num_burn=num_burn,
        calc_llh=False,
    )
    print("Run 3 - eta = 0.001, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

###################
# NON-EQUILIBRIUM #
###################
print("\n")

# setting up model
h_init = np.random.uniform(-1.5, 1.5, num_units)
J_init = np.random.normal(0, 1, (num_units, num_units))
J_init = (J_init.T + J_init) * np.sqrt(2) / 2
np.fill_diagonal(J_init, 0)

neq_model = ising.NeqModel(J_init, h_init)

# inference

neq_fitter = fitter.NeqFitter(neq_model)
print("Starting inference (NEQ)")
print("TAP")
t1 = time.time()
neq_fitter.TAP(sample)
t2 = time.time()
utils.print_elapsed_time(t1, t2)

print("Likelihood maximization")
t1 = time.time()
neq_fitter.maximize_likelihood(
    sample=sample,
    max_steps=4_000,
    learning_rate=0.005,
    win_size=win_size,
    tolerance=1e-16,
)
print("Run 1 - eta = 0.05, DONE")
t2 = time.time()
utils.print_elapsed_time(t1, t2)

if not is_testing:
    t1 = time.time()
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=4_000,
        learning_rate=0.0025,
        win_size=win_size,
        tolerance=1e-16,
    )
    print("Run 2 - eta = 0.025, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

    t1 = time.time()
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=10_000,
        learning_rate=0.001,
        win_size=win_size,
        tolerance=1e-16,
    )
    print("Run 3 - eta = 0.01, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

# storing data
print("\n")
params_path = "./data/params/"  # asuming we're in root
print("Storing data.")
params_data = {
    "eq_couplings": eq_model.getCouplings(),
    "eq_fields": eq_model.getFields(),
    "neq_couplings": neq_model.getCouplings(),
    "neq_fields": neq_model.getFields(),
}
np.save(params_path + f"params_{mouse_name}_{num_units}.npy", params_data)

# for notebook/other script:
# loaded_data = np.load('params_{mouse_name}_{num_units}.npy', allow_pickle=True).item()
