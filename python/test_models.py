import sys
path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)
import ising

import src.misc_plotting as plotting
import src.utils as utils
import src.model_eval as eval

import numpy as np
import time


##############
# SIMULATION #
##############

# setting up model
num_units = 10
num_sims = 15000
num_burn = 1000

beta = 1.3
h = np.random.uniform(-.3 * beta, .3 * beta, num_units)
J = np.random.normal(0,  beta / np.sqrt(num_units), (num_units, num_units))
for i in range(num_units):
    J[i, i] = 0
    for j in range(i+1, num_units):
        J[j, i] = J[i, j]

true_model = ising.EqModel(num_units, J, h)

# simulating
t0 = time.time()

true_sim = true_model.simulate(num_sims, num_burn)

t1 = time.time()
dt = t1 - t0
print("Simulation took {:.2f} seconds.".format(dt))


#############
# INFERENCE #
#############

# setting up model
h_init = np.random.uniform(-1.5, 1.5, num_units)
J_init = np.random.normal(0,  1,  (num_units, num_units))
J_init = (J_init.T + J_init) * np.sqrt(2) / 2

ml_model = ising.EqModel(num_units, J_init, h_init)

# inference
t0 = time.time()

lr = 0.01
max_steps = 100
ising.setMaxLikelihoodParamsEqModel(ml_model, true_sim, max_steps, lr, num_sims, num_burn)

# t1 = time.time()
# dt = t1 - t0
# print("Inference took {:.2f} seconds.".format(dt))

# plotting.scatter_compare_2d_obs(J, eq_model.getCouplings())

# rec_err = utils.get_reconstruction_err(J, eq_model.getCouplings())
# rmse = utils.get_rmse(J, eq_model.getCouplings())

# print(f"RMSE: {rmse}")
# print(f"Reconstructino error: {rec_err}")

#######################################################################################

def get_metadata(
    num_units,
    is_empirical_analysis,
    eq_inv_methods=[],
    neq_inv_methods=[],
    **kwargs
):
    metadata = {}
    metadata["num_units"] = num_units

    if is_empirical_analysis:
        bin_width = kwargs['bin_width']
        num_bins = kwargs['num_bins']
        metadata['bin_width'] = bin_width
    else:
        num_sims = kwargs['num_sims']
        num_burn = kwargs.get('num_burn', 1000)
        true_fields = kwargs['true_fields']
        true_couplings = kwargs['true_couplings']
        metadata["true_model"] = {
            'true_fields' : true_fields,
            'true_couplings' : true_couplings,
            'num_sims' : num_sims,
            'num_burn' : num_burn,
        }

    if (eq_inv_methods != []) or (neq_inv_methods != []):
        metadata["inverse_methods"] = {
            'EQ' : eq_inv_methods,
            'NEQ' : neq_inv_methods,
        }
        if ('ML' in eq_inv_methods) or ('ML' in neq_inv_methods):
            # each can be dict if multiple ML models with different hyperparams
            num_steps = kwargs['num_steps']
            learning_rate = kwargs['learning_rate']
            is_converged = kwargs['is_converged']
            metadata['maximum_likelihood'] = {
                'num_steps' : num_steps,
                'learning_rate' : learning_rate,
                'is_converged' : is_converged,
            }
        if ('ML' in eq_inv_methods):
            num_sims_ml = kwargs['num_sims_ml']
            num_burn_ml = kwargs.get('num_burn_ml', 1000)
            metadata['maximum_likelihood']['num_sims'] = num_sims_ml
            metadata['maximum_likelihood']['num_burn'] = num_burn_ml
    return metadata


labels = ["ML"]
metadata = get_metadata(
    num_units=num_units,
    is_empirical_analysis=False,
    eq_inv_methods=labels,
    num_sims=num_sims,
    true_fields="uniform(-.3 * beta, .3 * beta); beta=1.3",
    true_couplings="normal(0,  beta / sqrt(num_units)); symmetric, beta=1.3",
    num_steps=max_steps,
    learning_rate=lr,
    is_converged=None,
    num_sims_ml=num_sims,
    num_burn_ml=num_burn,
)


def get_analysis_path(analysis_name, num_units, bin_width):
    analysis_path = './analyses/'
    dir_name = f'n{num_units}b{bin_width}{analysis_name}'
    analysis_path += f'./{dir_name}/'
    return analysis_path


analysis_name = "test"
bin_width = 0
analysis_path = get_analysis_path(analysis_name, num_units, bin_width)

layout_spec = {
    ("fields", "histogram"): (1, 1),
    ("fields", "scatter"): (1, 2),
    ("couplings", "histogram"): (2, 1),
    ("couplings", "scatter"): (2, 2),
}

eval = eval.IsingEval(
    analysis_path=analysis_path, 
    metadata=metadata, 
    true_model=true_model, 
    est_models=[ml_model], 
    true_sample=true_sim,
    est_samples=[None], 
    labels=labels,
    layout_spec=layout_spec
)
eval.generate_plots()

#######################################################################################

# I think I might be able to use the IsingEval class from the previous iteration
# of this project without the need for too much changes (I think only get_ftr())

#########
# TO-DO #
#########

# Adapt model-eval to the C++ versions of the model and sample classes
# Adapt get_ftr()
# Should we keep the same structure as we had in python with metadata and plots?
# (might need to adjust paths or something in model_eval.py)

# Testing the MCMC sampler
# Derive expressions for the 1st and 2nd order moments with indep.-pair couplings,
# and make a function for plotting:
#   1) direct comparisons (scatter-plot)
#   2) error curves like we did before (or better)

# Implement the remaining inverse methods
#   - Pseudolikelihood maximization for the EQ model
#   - Mean-field methods
#   - etc.
#   - NEQ inverse methods
# Note: I'd like to reproduce some of the plots in Nguyen et al.

# Note2: Don't make a mess of the python implementation just because we're making a bigger 
# and badder version here: I'd like to be able to document some of the performance gains
# (e.g., in a blog post on my website)
