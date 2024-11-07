import sys
import os

path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising/build"
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
    "--models", type=str, help="Which models to use (EQ, NEQ, or Both).", default="Both"
)
parser.add_argument(
    "--mouse_name", type=str, help="Name of the recorded mouse.", default="Angie"
)
parser.add_argument("--num_units", type=int, help="Number of units.", default=10)
parser.add_argument("--num_sims", type=int, help="Name of simulations.", default=15_000)
parser.add_argument(
    "--use_prev_params",
    type=bool,
    help="Use previously inferred parameters if available.",
    default=False,
)
args = parser.parse_args()

# variable hyperparams
models = args.models
mouse_name = args.mouse_name
num_units = args.num_units
num_sims = args.num_sims
use_prev_params = args.use_prev_params

print(
    f"""
Running with:
- models: {models}
- use_prev_params: {use_prev_params}  
"""
)

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

# getting prev. params
print("\n")
params_path = "./data/params/"  # asuming we're in root
params_fname = params_path + f"params_{mouse_name}_{num_units}.npy"
if use_prev_params and os.path.exists(params_fname):
    print("Loading previous parameters.")
    params_data = np.load(params_fname, allow_pickle=True).item()
    eq_couplings = params_data["eq_couplings"]
    eq_fields = params_data["eq_fields"]

    neq_couplings = params_data["neq_couplings"]
    neq_fields = params_data["neq_fields"]


###############
# EQUILIBRIUM #
###############
print("\n")

# setting up model
if not use_prev_params:
    eq_fields = np.random.uniform(-1.5, 1.5, num_units)
    eq_couplings = np.random.normal(0, 1, (num_units, num_units))
    eq_couplings = (eq_couplings.T + eq_couplings) * np.sqrt(2) / 2
    np.fill_diagonal(eq_couplings, 0)

eq_model = ising.EqModel(eq_couplings, eq_fields)

if models == "EQ" or models == "Both":
    # inference
    eq_fitter = fitter.EqFitter(eq_model)

    print("Starting inference (EQ)")
    # if not use_prev_params:
    #     print("TAP")
    #     t1 = time.time()
    #     eq_fitter.TAP(sample)
    #     t2 = time.time()
    #     utils.print_elapsed_time(t1, t2)

    # t1 = time.time()
    # eq_fitter.maximize_likelihood(
    #     sample=sample,
    #     max_steps=1_000,
    #     learning_rate=0.0025,
    #     win_size=win_size,
    #     tolerance=tol_ml,
    #     num_sims=num_sims,
    #     num_burn=num_burn,
    #     calc_llh=False,
    # )
    # print("Run - eta = 0.0025, DONE")
    # t2 = time.time()
    # utils.print_elapsed_time(t1, t2)

    # t1 = time.time()
    # eq_fitter.maximize_likelihood(
    #     sample=sample,
    #     max_steps=10_000,
    #     learning_rate=0.001,
    #     win_size=win_size,
    #     tolerance=tol_ml,
    #     num_sims=num_sims,
    #     num_burn=num_burn,
    #     calc_llh=False,
    # )
    # print("Run - eta = 0.001, DONE")
    # t2 = time.time()
    # utils.print_elapsed_time(t1, t2)

    t1 = time.time()
    eq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=5_000,
        learning_rate=0.00001,
        win_size=win_size,
        tolerance=tol_ml,
        num_sims=num_sims,
        num_burn=num_burn,
        calc_llh=False,
    )
    print("Run - eta = 0.00001, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

###################
# NON-EQUILIBRIUM #
###################
print("\n")

# setting up model
if not use_prev_params:
    neq_fields = np.random.uniform(-1.5, 1.5, num_units)
    neq_couplings = np.random.normal(0, 1, (num_units, num_units))
    J_init = (neq_couplings.T + neq_couplings) * np.sqrt(2) / 2
    np.fill_diagonal(neq_couplings, 0)

neq_model = ising.NeqModel(neq_couplings, neq_fields)

if models == "NEQ" or models == "Both":
    # inference
    neq_fitter = fitter.NeqFitter(neq_model)
    print("Starting inference (NEQ)")
    # if not use_prev_params:
    #     print("TAP")
    #     t1 = time.time()
    #     neq_fitter.TAP(sample)
    #     t2 = time.time()
    #     utils.print_elapsed_time(t1, t2)

    # t1 = time.time()
    # neq_fitter.maximize_likelihood(
    #     sample=sample,
    #     max_steps=1_000,
    #     learning_rate=0.0025,
    #     win_size=win_size,
    #     tolerance=1e-16,
    # )
    # print("Run - eta = 0.0025, DONE")
    # t2 = time.time()
    # utils.print_elapsed_time(t1, t2)

    # t1 = time.time()
    # neq_fitter.maximize_likelihood(
    #     sample=sample,
    #     max_steps=10_000,
    #     learning_rate=0.001,
    #     win_size=win_size,
    #     tolerance=1e-16,
    # )
    # print("Run - eta = 0.001, DONE")
    # t2 = time.time()
    # utils.print_elapsed_time(t1, t2)

    t1 = time.time()
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=5_000,
        learning_rate=0.0005,
        win_size=win_size,
        tolerance=1e-16,
    )
    print("Run - eta = 0.0005, DONE")
    t2 = time.time()
    utils.print_elapsed_time(t1, t2)

    # storing data
    print("Storing data.")
    params_data = {
        "eq_couplings": eq_model.getCouplings(),
        "eq_fields": eq_model.getFields(),
        "neq_couplings": neq_model.getCouplings(),
        "neq_fields": neq_model.getFields(),
    }
    np.save(params_fname, params_data)

# for notebook/other script:
# loaded_data = np.load('params_{mouse_name}_{num_units}.npy', allow_pickle=True).item()
