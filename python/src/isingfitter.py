import sys
path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)
import ising
import numpy as np


class IsingFitter:
    def __init__(self, model):
        self.model = model
        #self.fields_history = []
        #self.couplings_history = []
        #self.llhs = []

    def check_sample_dims(self, sample):
        if self.model.getNumUnits() != sample.getNumUnits():
            raise ValueError("Shape mismatch between model and sample.")

    def maximize_likelihood(self, sample, max_steps, learning_rate, num_sims, num_burn):
        raise NotImplementedError(
            "Not implemented in base class: use EqFitter or NeqFitter."
        )
    
    def naive_mean_field(self, sample):
        self.check_sample_dims(sample)

        couplings_est = self._get_nmf_couplings(sample)
        self.model.couplings = couplings_est

        # couplings must be set before calling _get_nmf_fields()
        fields_est = self._get_nmf_fields(sample)
        self.model.fields = fields_est

    def _get_nmf_couplings(self, sample):
        raise NotImplementedError(
            "Not implemented in base class: use EqFitter or NeqFitter."
        )


class EqFitter(IsingFitter):
    def __init__(self, model):
        super().__init__(model)
        # might do a check, like: 
        # if not isinstance(model, ising.EqModel):
        #     raise ValueError("Model must be an equilibrium Ising model (EqModel).")

    def maximize_likelihood(self, sample, max_steps, learning_rate, num_sims, num_burn):
        ising.setMaxLikelihoodParamsEq(self.model, sample, max_steps, learning_rate, num_sims, num_burn)
        # Note: might use **kwargs like original implementation