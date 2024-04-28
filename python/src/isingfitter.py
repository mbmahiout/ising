import sys
path2cpp_pkg = "/Users/mariusmahiout/Documents/repos/ising_core/build"
sys.path.append(path2cpp_pkg)
import ising

from src.utils import stable_arctanh, get_inv_mat

import numpy as np

class IsingFitter:
    def __init__(self, model):
        self.model = model

        # parameters history
        self.fields_history = []
        self.couplings_history = []

        # gradients
        self.fields_grads = []
        self.couplings_grads = []

        # parameter statistics
        self.av_fields = []
        self.av_couplings = []

        self.sd_fields = []
        self.sd_couplings = []

        self.min_fields = []
        self.min_couplings = []

        self.max_fields = []
        self.max_couplings = []

        # optionally, for testing
        self.llhs = []

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
        self.model.setCouplings(couplings_est)

        # couplings must be set before calling _get_nmf_fields()
        fields_est = self._get_nmf_fields(sample)
        self.model.setFields(fields_est)

    def TAP(self, sample):
        self.check_sample_dims(sample)
        couplings_est = self._get_TAP_couplings(sample)
        self.model.setCouplings(couplings_est)

        fields_est = self._get_TAP_fields(sample)
        self.model.setFields(fields_est)

    def _get_nmf_couplings(self, sample):
        raise NotImplementedError(
            "Not implemented in base class: use EqFitter or NeqFitter."
        )

    def _get_nmf_fields(self, sample):
        means = sample.getMeans()
        fields_est = stable_arctanh(means) - np.matmul(self.model.getCouplings(), means)
        return fields_est

    def _get_TAP_couplings(self, sample):
        raise NotImplementedError(
            "Not implemented in base class: use EqFitter or NeqFitter."
        )

    def _get_TAP_fields(self, sample):
        fields_nmf = self._get_nmf_fields(sample)
        onsager_terms = self._get_onsager_terms(sample)
        fields_est = fields_nmf + onsager_terms
        return fields_est

    def _get_ccorrs_inv(self, sample):
        ccorrs = sample.getConnectedCorrs()
        num_units = self.model.getNumUnits()
        ccorrs_inv = get_inv_mat(ccorrs, size=num_units)
        return ccorrs_inv

    def _get_A_naive(self, sample):
        # A_naive is the diagonal matrix whose ith diagonal element is 1 - mi²
        means = sample.getMeans()
        means_sq = np.square(means)
        I = np.identity(self.model.getNumUnits())
        means_sq = I * means_sq
        A_naive = I - means_sq
        return A_naive

    def _get_onsager_terms(self, sample):
        I = np.identity(self.model.getNumUnits())
        # here, we assume the couplings have been set
        couplings_sq = np.square(self.model.getCouplings())
        np.fill_diagonal(couplings_sq, 0)
        A = self._get_A_naive(sample)
        A_diag = np.diagonal(A)
        onsager_sums = np.matmul(couplings_sq, A_diag)
        means = sample.getMeans()
        means_mat = I * means
        onsager_terms = np.matmul(means_mat, onsager_sums)
        return onsager_terms


class EqFitter(IsingFitter):
    def __init__(self, model):
        super().__init__(model)
        # might do a check, like:
        # if not isinstance(model, ising.EqModel):
        #     raise ValueError("Model must be an equilibrium Ising model (EqModel).")

    def maximize_likelihood(
            self,
            sample,
            max_steps,
            learning_rate=0.1,
            use_adam=True,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-5,
            win_size=10,
            tolerance=1e-5,
            num_sims=0,
            num_burn=0,
            calc_llh=False
    ):
        out = ising.gradientAscentEQ(
            self.model,
            sample,
            max_steps,
            learning_rate,
            use_adam,
            beta1,
            beta2,
            epsilon,
            win_size,
            tolerance,
            num_sims,
            num_burn,
            calc_llh
        )
        # parameters history
        self.fields_history = out.params.fields
        self.couplings_history = out.params.couplings

        # gradients
        self.fields_grads = out.grads.fieldsGrads
        self.couplings_grads = out.grads.couplingsGrads

        # parameter statistics
        self.av_fields = out.stats.avFields
        self.av_couplings = out.stats.avCouplings

        self.sd_fields = out.stats.sdFields
        self.sd_couplings = out.stats.sdCouplings

        self.min_fields = out.stats.minFields
        self.min_couplings = out.stats.minCouplings

        self.max_fields = out.stats.maxFields
        self.max_couplings = out.stats.maxCouplings

        if calc_llh:
            self.llhs = out.stats.LLHs

    def _get_nmf_couplings(self, sample):
        ccorrs_inv = self._get_ccorrs_inv(sample)
        couplings_est = -ccorrs_inv
        np.fill_diagonal(couplings_est, 0)  # remove possible self-couplings
        return couplings_est

    def _get_TAP_couplings(self, sample):
        def _get_min_root(ccorrs_inv, means, i, j):
            sqrt_fact = np.lib.scimath.sqrt(
                1 - 8 * ccorrs_inv[i, j] * means[i] * means[j]
            )
            root1 = (-1 + sqrt_fact) / (4 * means[i] * means[j])
            root2 = (-1 - sqrt_fact) / (4 * means[i] * means[j])
            roots = [root1, root2]
            real_roots = list(filter(lambda r: abs(r.imag) < 1e-5, roots))
            if not len(real_roots) == 0:
                real_roots = list(map(lambda r: r.real, real_roots))
            else:
                real_roots = real_roots = list(map(lambda r: r.real, roots))
            min_root = min(real_roots, key=abs)
            return min_root

        num_units = self.model.getNumUnits()
        ccorrs_inv = self._get_ccorrs_inv(sample)
        couplings_est = np.zeros((num_units, num_units))
        means = sample.getMeans()
        for i in range(num_units):
            for j in range(i + 1, num_units):
                min_root = _get_min_root(ccorrs_inv, means, i, j)
                couplings_est[i, j] = min_root
                couplings_est[j, i] = couplings_est[i, j]
        return couplings_est
