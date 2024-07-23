import sys
import os

path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)

import ising
import os
import sys

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


def single_run(
    num_units,
    fname,
    mouse_name,
    bin_width=50,
    lr=0.1,
    win_size=10,
    max_steps=5000,
):
    sample = pre.get_recording_sample(
        fname=fname,
        mouse_name=mouse_name,
        bin_width=bin_width,
        num_units=num_units,
    )

    # setting up model
    h_init = np.random.uniform(-1.5, 1.5, num_units)
    J_init = np.random.normal(0, 1, (num_units, num_units))
    J_init = (J_init.T + J_init) * np.sqrt(2) / 2
    np.fill_diagonal(J_init, 0)
    neq_model = ising.NeqModel(J_init, h_init)

    # fitting model
    neq_fitter = fitter.NeqFitter(neq_model)
    neq_fitter.TAP(sample)

    # inference
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=4000,
        learning_rate=0.05,
        win_size=win_size,
        tolerance=1e-16,
    )
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=6000,
        learning_rate=0.025,
        win_size=win_size,
        tolerance=1e-16,
    )
    neq_fitter.maximize_likelihood(
        sample=sample,
        max_steps=8000,
        learning_rate=0.01,
        win_size=win_size,
        tolerance=1e-16,
    )
    return neq_model.getCouplings(), neq_model.getFields()


mouse_name = "Angie"
fname = "RESULTS_Angie_20170825_1220_allbeh_1000s.mat"
path = "analyses"
utils.make_dir(path)

bin_width = 50  # ms

size_range = [5, 6, 7, 8]  # , 90, 110, 130, 150]
couplings = dict((num_units, None) for num_units in size_range)
fields = dict((num_units, None) for num_units in size_range)
for num_units in size_range:
    J, h = single_run(num_units, fname, mouse_name)
    couplings[num_units] = J
    fields[num_units] = h
    print(f"{num_units} units - DONE")
