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


mouse_name = "Angie"
sens_param = "time"

path = "analyses"
utils.make_dir(path)

num_units = 35  # 0
bin_width = 50  # ms

sample = pre.get_recording_sample(
    fname="RESULTS_Angie_20170825_1220_allbeh_1000s.mat",
    mouse_name=mouse_name,
    bin_width=bin_width,
    num_units=num_units,
)

num_sims = 15_000  # 50_000
num_burn = 5_000
lr = 0.1
win_size = 10
tol_ml = 1e-3
tol_pl = 1e-9
max_steps = 5000

###############
# EQUILIBRIUM #
###############

# setting up model
h_init = np.random.uniform(-1.5, 1.5, num_units)
J_init = np.random.normal(0, 1, (num_units, num_units))
J_init = (J_init.T + J_init) * np.sqrt(2) / 2
np.fill_diagonal(J_init, 0)

eq_model = ising.EqModel(J_init, h_init)

eq_fitter = fitter.EqFitter(eq_model)
eq_fitter.TAP(sample)

# inference

print("Starting inference (EQ)")
eq_fitter.maximize_likelihood(
    sample=sample,
    max_steps=4_00,  # 0,
    learning_rate=0.05,
    win_size=win_size,
    tolerance=tol_ml,
    num_sims=num_sims,
    num_burn=num_burn,
    calc_llh=False,
)
print("Run 1 - eta = 0.05, DONE")

# eq_fitter.maximize_likelihood(
#     sample=sample,
#     max_steps=4_000,
#     learning_rate=0.025,
#     win_size=win_size,
#     tolerance=tol_ml,
#     num_sims=num_sims,
#     num_burn=num_burn,
#     calc_llh=False,
# )
# print("Run 2 - eta = 0.025, DONE")

# eq_fitter.maximize_likelihood(
#     sample=sample,
#     max_steps=8_000,
#     learning_rate=0.01,
#     win_size=win_size,
#     tolerance=tol_ml,
#     num_sims=num_sims,
#     num_burn=num_burn,
#     calc_llh=False,
# )
# print("Run 3 - eta = 0.01, DONE")


###################
# NON-EQUILIBRIUM #
###################

# setting up model
h_init = np.random.uniform(-1.5, 1.5, num_units)
J_init = np.random.normal(0, 1, (num_units, num_units))
J_init = (J_init.T + J_init) * np.sqrt(2) / 2
np.fill_diagonal(J_init, 0)

neq_model = ising.NeqModel(J_init, h_init)

neq_fitter = fitter.NeqFitter(neq_model)
neq_fitter.TAP(sample)

# inference
print("Starting inference (NEQ)")
neq_fitter.maximize_likelihood(
    sample=sample,
    max_steps=4_00,  # 0,
    learning_rate=0.05,
    win_size=win_size,
    tolerance=1e-16,
)
print("Run 1 - eta = 0.05, DONE")

# neq_fitter.maximize_likelihood(
#     sample=sample,
#     max_steps=4_000,
#     learning_rate=0.025,
#     win_size=win_size,
#     tolerance=1e-16,
# )
# print("Run 2 - eta = 0.025, DONE")

# neq_fitter.maximize_likelihood(
#     sample=sample,
#     max_steps=8_000,
#     learning_rate=0.01,
#     win_size=win_size,
#     tolerance=1e-16,
# )
# print("Run 3 - eta = 0.01, DONE")

eq_sim = eq_model.simulate(num_sims, num_burn)
neq_sim = neq_model.simulate(num_sims, num_burn)

labels = ["EQ", "NEQ"]
metadata = utils.get_metadata(  # gotta change this
    num_units=num_units,
    is_empirical_analysis=False,
    eq_inv_methods=labels,
    num_sims=num_sims,
    true_fields="Angie performing 1",
    true_couplings="Angie performing 1",
    num_steps=max_steps,
    learning_rate=lr,
    is_converged=None,
    num_sims_ml=num_sims,
    num_burn_ml=num_burn,
)


analysis_name = "rec_test"
bin_width = 0
analysis_path = utils.get_analysis_path(analysis_name, num_units, bin_width)

layout_spec1 = {
    ("fields", "histogram"): (1, 1),
    ("couplings", "histogram"): (1, 2),
}
ising_eval1 = eval.IsingEval(
    analysis_path=analysis_path,
    metadata=metadata,
    true_model=eq_model,
    est_models=[neq_model],  # [eq_model, neq_model],  #
    true_sample=sample,
    est_samples=[neq_sim],  # [eq_sim, neq_sim],
    labels=["NEQ"],
    layout_spec=layout_spec1,
)
ising_eval1.generate_plots(is_pdf=False)

layout_spec2 = {
    ("means", "scatter"): (1, 1),
    ("ccorrs", "scatter"): (1, 2),
    ("dcorrs", "scatter"): (2, 1),
    ("tricorrs", "scatter"): (2, 2),
}
ising_eval2 = eval.IsingEval(
    analysis_path=analysis_path,
    metadata=metadata,
    true_model=None,
    est_models=[eq_model, neq_model],  #
    true_sample=sample,
    est_samples=[eq_sim, neq_sim],
    labels=labels,
    layout_spec=layout_spec2,
)
ising_eval2.generate_plots(is_pdf=False)


units2show = 50
units2show = min(num_units, units2show)
misc_plotting.plot_coupling_graph_and_matrix(
    J=neq_model.getCouplings(),
    threshold=0.0,
    num_units=units2show,
    directed=True,
    fname=f"{analysis_path}/coupling_graph_neq{num_units}_{units2show}",
)
misc_plotting.plot_coupling_graph_and_matrix(
    J=eq_model.getCouplings(),
    threshold=0.0,
    num_units=units2show,
    directed=False,
    fname=f"{analysis_path}/coupling_graph_eq{num_units}_{units2show}",
)

# if units2show != num_units:
#     misc_plotting.plot_coupling_graph_and_matrix(
#         J=neq_model.getCouplings(),
#         threshold=0.0,
#         num_units=num_units,
#         directed=True,
#         fname=f"{analysis_path}/coupling_graph_neq_{num_units}_{num_units}",
#     )
#     misc_plotting.plot_coupling_graph_and_matrix(
#         J=eq_model.getCouplings(),
#         threshold=0.0,
#         num_units=num_units,
#         directed=False,
#         fname=f"{analysis_path}/coupling_graph_eq{num_units}_{num_units}",
#     )
