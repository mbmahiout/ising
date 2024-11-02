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
import pandas as pd
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from IPython.display import display, HTML
from ipywidgets import HBox, VBox, widgets
import scipy
import pandas as pd

data_folder = "./data/"
mouse_name = "Angie"


def load_params(num_units: int, data_folder: str, mouse_name: str):
    return np.load(
        data_folder + "params/" + f"params_{mouse_name}_{num_units}.npy",
        allow_pickle=True,
    ).item()


num_units_range = range(10, 270 + 10, 10)

params = dict(
    (num_units, load_params(num_units, data_folder, mouse_name))
    for num_units in num_units_range
)

# EQ params
eq_couplings = dict(
    (num_units, params[num_units]["eq_couplings"]) for num_units in num_units_range
)
eq_fields = dict(
    (num_units, params[num_units]["eq_fields"]) for num_units in num_units_range
)

# NEQ params
neq_couplings = dict(
    (num_units, params[num_units]["neq_couplings"]) for num_units in num_units_range
)
neq_fields = dict(
    (num_units, params[num_units]["neq_fields"]) for num_units in num_units_range
)

num_units = 100
bin_width = 50  # ms
num_sims = 50_000
num_burn = 5000

mouse_name = "Angie"
# angie performing 1
recording_fname = "RESULTS_Angie_20170825_1220_allbeh_1000s.mat"

sample = pre.get_recording_sample(
    fname=recording_fname,
    mouse_name=mouse_name,
    bin_width=bin_width,
    num_units=num_units,
)

J_eq = eq_couplings[num_units]
h_eq = eq_fields[num_units]

J_neq = neq_couplings[num_units]
h_neq = neq_fields[num_units]

eq_model = ising.EqModel(J_eq, h_eq)
neq_model = ising.NeqModel(J_neq, h_neq)


eq_sample = eq_model.simulate(num_sims, num_burn)
neq_sample = neq_model.simulate(num_sims, num_burn)

labels = ["EQ", "NEQ"]
metadata = utils.get_metadata(
    num_units=num_units,
    is_empirical_analysis=True,
    bin_width=bin_width,
    num_bins=sample.getNumBins(),
    eq_inv_methods=["ML"],
    neq_inv_methods=["ML"],
    num_steps="...",
    learning_rate="...",
    is_converged="...",
    num_sims_ml=num_sims,
    num_burn_ml=num_burn,
)


analysis_name = "recording_analysis"
bin_width = 0
analysis_path = utils.get_analysis_path(analysis_name, num_units, bin_width)


layout_spec = {("tricorrs", "scatter"): (1, 1)}

ising_eval = eval.IsingEval(
    analysis_path=analysis_path,
    metadata=metadata,
    true_model=None,
    est_models=[eq_model, neq_model],
    true_sample=sample,
    est_samples=[eq_sample, neq_sample],
    labels=labels,
    layout_spec=layout_spec,
    is_eq_vs_neq=True,
)
ising_eval.generate_plots(is_pdf=False)
