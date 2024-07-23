import sys

path2cpp_pkg = (
    "/Users/mariusmahiout/Documents/repos/ising_core/build"  # change to relative import
)
sys.path.append(path2cpp_pkg)
import ising

from utils import stable_arctanh, get_inv_mat
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

    def set_gradient_ascent_outputs(self, out, calc_llh):
        # parameters history
        self.fields_history += out.params.fields
        self.couplings_history += out.params.couplings

        # gradients
        self.fields_grads += out.grads.fieldsGrads
        self.couplings_grads += out.grads.couplingsGrads

        # parameter statistics
        self.av_fields += out.stats.avFields
        self.av_couplings += out.stats.avCouplings

        self.sd_fields += out.stats.sdFields
        self.sd_couplings += out.stats.sdCouplings

        self.min_fields += out.stats.minFields
        self.min_couplings += out.stats.minCouplings

        self.max_fields += out.stats.maxFields
        self.max_couplings += out.stats.maxCouplings

        if calc_llh:
            self.llhs += out.stats.LLHs

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
        if not isinstance(model, ising.EqModel):
            raise ValueError("Model must be an equilibrium Ising model (EqModel).")
        super().__init__(model)

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
        calc_llh=False,
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
            calc_llh,
        )
        self.set_gradient_ascent_outputs(out, calc_llh)

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
            min_root = min(roots, key=abs)
            return min_root.real

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


class NeqFitter(IsingFitter):
    def __init__(self, model, dt=1):
        if not isinstance(model, ising.NeqModel):
            raise ValueError("Model must be a non-equilibrium Ising model (NeqModel).")
        super().__init__(model)
        self.dt = dt

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
        calc_llh=False,
    ):
        out = ising.gradientAscentNEQ(
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
            calc_llh,
        )
        self.set_gradient_ascent_outputs(out, calc_llh)

    def _calc_F(self, A_naive, dcorrs, ccorrs_inv):
        num_units = self.model.getNumUnits()
        A_naive_inv = get_inv_mat(A_naive, size=num_units)
        couplings_nmf = self._get_neq_mean_field_couplings(
            A_naive_inv, dcorrs, ccorrs_inv
        )
        couplings_nmf_sq = couplings_nmf**2
        A_naive_diag = np.diagonal(A_naive)
        coef0_vec = -A_naive_diag * np.matmul(couplings_nmf_sq, A_naive_diag)

        min_roots = []
        for i in range(num_units):
            roots = np.roots([1, -2, 1, coef0_vec[i]])
            min_root = min(roots, key=abs)
            min_roots.append(min_root.real)
        F = np.array(min_roots)
        return F

    def _get_neq_mean_field_couplings(self, A_inv, dcorrs, ccorrs_inv):
        couplings_est = np.matmul(np.matmul(A_inv, dcorrs), ccorrs_inv)
        np.fill_diagonal(couplings_est, 0)
        return couplings_est

    def _get_nmf_couplings(self, sample):
        # while the kinetic Ising model does allow for non-zero self-couplings,
        # we are setting them to zero for the sake of simplicity

        num_units = self.model.getNumUnits()
        A = self._get_A_naive(sample)
        A_inv = get_inv_mat(A, size=num_units)
        dcorrs = sample.getDelayedCorrs(self.dt)
        ccorrs_inv = self._get_ccorrs_inv(sample)

        couplings_est = self._get_neq_mean_field_couplings(A_inv, dcorrs, ccorrs_inv)
        return couplings_est

    def _get_TAP_couplings(self, sample):
        num_units = self.model.getNumUnits()
        A_naive = self._get_A_naive(sample)
        dcorrs = sample.getDelayedCorrs(self.dt)
        ccorrs_inv = self._get_ccorrs_inv(sample)

        F = self._calc_F(A_naive, dcorrs, ccorrs_inv)
        A = A_naive * (np.ones(num_units) - F)
        A_inv = get_inv_mat(A, size=num_units)

        couplings_est = self._get_neq_mean_field_couplings(A_inv, dcorrs, ccorrs_inv)
        return couplings_est
