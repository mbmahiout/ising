import numpy as np
from scipy.stats import ks_2samp
import yaml
import re
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
from ipywidgets import HBox, VBox, widgets
from IPython.display import display
import os

from src.utils import (
    get_rmse,
    get_3rd_order_corrs,
    get_unique_3d_tensor_vals,
    get_unique_matrix_vals,
)


class IsingEval:
    def __init__(
        self,
        true_model,
        est_models,
        true_sample,
        est_samples,
        labels,
        layout_spec,
        analysis_path=None,
        metadata=None,
    ):
        """
        layout_spec: A dictionary specifying the layout of plots and their types. Example:
        {("fields", "scatter"): (1, 1), ("means", "distribution"): (1, 2), ("couplings", "scatter"): (2, 1), ...}
        """
        self.true_model = true_model
        self.est_models = est_models
        self.true_sample = true_sample
        self.est_samples = est_samples
        self.labels = labels
        self.layout_spec = layout_spec
        self.analysis_path = analysis_path
        self.metadata = metadata
        self.num_rows = max([spec[0] for spec in layout_spec.values()])
        self.num_cols = max([spec[1] for spec in layout_spec.values()])
        self.save_button_figure = widgets.Button(description="Save figure")
        self.save_button_figure.on_click(self.save_figure)
        self.save_button_metadata = widgets.Button(description="Save metadata")
        self.save_button_metadata.on_click(self.save_metadata)
        self.save_button_results = widgets.Button(description="Save results")
        self.save_button_results.on_click(self.save_results)
        self.color_palette = [
            "blue",
            "green",
            "red",
            "orange",
            "purple",
            "cyan",
            "brown",
            "magenta",
        ]

    ################
    # core methods #
    ################

    def generate_plots(self):
        self.fig = go.FigureWidget(
            make_subplots(
                rows=self.num_rows,
                cols=self.num_cols,
                subplot_titles=None,
                horizontal_spacing=0.15,
                vertical_spacing=0.15,
            )
        )

        for (ftr_name, plot_type), (row, col) in self.layout_spec.items():
            if plot_type == "scatter":
                self.plot_scatter(ftr_name, row, col)
            elif plot_type == "histogram":

                self.plot_histogram(ftr_name, row, col)

        self.fig.update_layout(barmode="overlay")
        self.fig.update_layout(bargap=0.0)
        self.fig.update_layout(
            height=400 * self.num_rows,
            width=400 * self.num_cols,
        )

        display(
            VBox(
                [
                    self.fig,
                    HBox(
                        [
                            self.save_button_figure,
                            self.save_button_results,
                            self.save_button_metadata,
                        ]
                    ),
                ]
            )
        )

    def save_figure(self, button):
        analysis_path = self.analysis_path
        figs_path = analysis_path + "figures/"
        IsingEval._make_dir(analysis_path)
        IsingEval._make_dir(figs_path)

        fig_fname = "fig1.pdf"
        curr_fig_path = figs_path + fig_fname

        path_exists = os.path.exists(curr_fig_path)
        while path_exists:
            curr_fig_path = IsingEval.increment_path_num(curr_fig_path)
            path_exists = os.path.exists(curr_fig_path)

        pio.write_image(self.fig, curr_fig_path)

    def save_results(self, button):  # make sure we don't overwrite
        analysis_path = self.analysis_path
        res_path = analysis_path + "results/"
        IsingEval._make_dir(analysis_path)
        IsingEval._make_dir(res_path)

        results = self.get_results()
        res_fname = "res1.yaml"
        curr_res_path = res_path + res_fname

        path_exists = os.path.exists(curr_res_path)
        while path_exists:
            curr_res_path = IsingEval.increment_path_num(curr_res_path)
            path_exists = os.path.exists(curr_res_path)

        with open(curr_res_path, "w") as file:
            yaml.dump(results, file, default_flow_style=False, sort_keys=False)

    def save_metadata(self, button):  # can overwrite (or only save once)
        analysis_path = self.analysis_path
        IsingEval._make_dir(analysis_path)

        metadata_fname = "metadata.yaml"
        with open(analysis_path + metadata_fname, "w") as file:
            yaml.dump(self.metadata, file, default_flow_style=False, sort_keys=False)

    #####################
    # auxiliary methods #
    #####################

    # --- plotting --- #

    def plot_scatter(self, ftr_name, row, col, pad=0.5):

        true_data = IsingEval.get_ftr(self.true_model, self.true_sample, ftr_name)
        est_datas = [
            IsingEval.get_ftr(est_model, est_sample, ftr_name)
            for est_model, est_sample in zip(self.est_models, self.est_samples)
        ]

        colors = self.color_palette[: len(self.labels)]
        for est_data, label, color in zip(est_datas, self.labels, colors):
            IsingEval._add_comparison_scatter(
                self.fig, true_data, est_data, color, label, row, col
            )

        # all_data = np.concatenate(est_datas + [true_data])
        min_val, max_val = np.min(true_data), np.max(true_data)
        interval = IsingEval.get_padded_interval(min_val, max_val, pad)

        IsingEval._add_id_line(self.fig, interval, row, col)

        ##########################################################################################

        # self.fig.update_xaxes(
        #     title_text="True", row=row, col=col
        # )
        # self.fig.update_yaxes(
        #     title_text="Estimated", row=row, col=col
        # )

        ftr_symbol = IsingEval.get_ftr_symbol(ftr_name)

        self.fig.update_xaxes(
            title_text=rf"$\Large {{{ftr_symbol}}}^{{\text{{True}}}}$",
            row=row,
            col=col,
        )
        self.fig.update_yaxes(
            title_text=rf"$\Large {{{ftr_symbol}}}^{{\text{{Est}}}}$",
            row=row,
            col=col,
        )

    ##########################################################################################

    def plot_histogram(self, ftr_name, row, col):
        num_bins = 20

        true_data = IsingEval.get_ftr(self.true_model, self.true_sample, ftr_name)
        est_datas = [
            IsingEval.get_ftr(est_model, est_sample, ftr_name)
            for est_model, est_sample in zip(self.est_models, self.est_samples)
        ]

        all_data = np.concatenate(est_datas + [true_data])
        min_val, max_val = np.min(all_data), np.max(all_data)

        bin_width = (max_val - min_val) / num_bins
        bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        colors = self.color_palette[: len(self.labels)]
        for est_data, label, color in zip(est_datas, self.labels, colors):
            counts, _ = np.histogram(est_data, bins=bin_edges, density=True)
            IsingEval._add_histogram(
                self.fig, counts, bin_centers, bin_width, label, color, row, col
            )
        counts, _ = np.histogram(true_data, bins=bin_edges, density=True)
        IsingEval._add_histogram(
            self.fig,
            counts,
            bin_centers,
            bin_width,
            label="True",
            color="grey",
            row=row,
            col=col,
        )

        self.fig.update_yaxes(range=[0, 1], row=row, col=col)
        self.fig.update_xaxes(
            title_text="Values", row=row, col=col
        )  # TO-DO: title_text=rf"${ftr_sumbol}$"
        self.fig.update_yaxes(title_text="Rel. Freq.", row=row, col=col)

    @staticmethod
    def _add_comparison_scatter(fig, true_data, est_data, color, label, row, col):
        showlegend = row == 1 and col == 1
        fig.add_trace(
            go.Scatter(
                x=true_data,
                y=est_data,
                mode="markers",
                marker=dict(color=color, size=7.5, symbol="circle"),
                opacity=0.75,
                name=label,
                legendgroup=label,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def _add_id_line(fig, interval, row, col):
        fig.add_trace(
            go.Scatter(
                x=interval,
                y=interval,
                mode="lines",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def _add_histogram(fig, counts, bin_centers, bin_width, label, color, row, col):
        showlegend = row == 1 and col == 1
        adjusted_counts = counts * bin_width
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=adjusted_counts,
                width=bin_width,
                marker=dict(color=color, opacity=0.75),
                name=label,
                legendgroup=label,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    # --- calculations --- #

    def get_results(self):
        results = {
            "scatters": self.get_scatter_results(),
            "histograms": self.get_hist_results(),
        }
        return results

    def get_scatter_results(self):
        scatter_results = {}
        for (ftr_name, plot_type), (row, col) in self.layout_spec.items():
            if plot_type == "scatter":
                scatter_results[f"{ftr_name} - ({row, col})"] = {
                    label: self.get_model_ftr_rmse(ftr_name, model, sample)
                    for label, model, sample in zip(
                        self.labels, self.est_models, self.est_samples
                    )
                }
        return scatter_results

    def get_hist_results(self):
        hist_results = {}

        for (ftr_name, plot_type), (row, col) in self.layout_spec.items():
            if plot_type == "histogram":
                hist_results[f"{ftr_name} - ({row, col})"] = self.get_ks_test_results(
                    ftr_name
                )
        return hist_results

    def get_model_ftr_rmse(self, ftr_name, model, sample):
        true_data = IsingEval.get_ftr(self.true_model, self.true_sample, ftr_name)
        est_data = IsingEval.get_ftr(model, sample, ftr_name)
        rmse = get_rmse(true_data.flatten(), est_data.flatten())
        return round(rmse.item(), 3)

    def get_ks_test_results(self, ftr_name):

        test_results = {}

        models = [self.true_model] + self.est_models
        samples = [self.true_sample] + self.est_samples
        labels = ["True"] + self.labels

        num_models = len(models)
        num_samples = len(samples)
        num_labels = len(labels)

        for m1, s1, l1 in zip(range(num_models), range(num_samples), range(num_labels)):
            for m2, s2, l2 in zip(
                range(m1 + 1, num_models),
                range(s1 + 1, num_samples),
                range(l1 + 1, num_labels),
            ):
                data1 = IsingEval.get_ftr(models[m1], samples[s1], ftr_name)
                data2 = IsingEval.get_ftr(models[m2], samples[s2], ftr_name)

                stat, pval = ks_2samp(data1, data2)
                if isinstance(stat, np.generic):
                    stat = stat.item()
                if isinstance(pval, np.generic):
                    pval = pval.item()
                label = f"{labels[l1]} & {labels[l2]}"
                test_results[label] = {"stat": round(stat, 3), "pval": round(pval, 3)}
        return test_results

    # --- misc --- #
    @staticmethod
    def get_ftr_symbol(ftr_name):
        # Define a mapping from ftr_name to LaTeX symbols
        ftr_symbols = {
            "fields": r"h_i",
            "couplings": r"J_{ij}",
            "means": r"m_i",
            "pcorrs": r"\chi_{ij}",
            "ccorrs": r"C_{ij}",
            "dcorrs": r"D_{ij}",
            "tricorrs": r"T_{ij}",
            "num-distr": r"p(N)",
        }
        return ftr_symbols.get(ftr_name, r"O")

    @staticmethod
    def get_ftr(model, sample, ftr_name):
        if ftr_name == "fields":
            return model.getFields()

        elif ftr_name == "couplings":
            return get_unique_matrix_vals(model.getCouplings())

        elif ftr_name == "means":
            return sample.getMeans()

        elif ftr_name == "pcorrs":
            return get_unique_matrix_vals(sample.getPairwiseCorrs())

        elif ftr_name == "ccorrs":
            return get_unique_matrix_vals(sample.getConnectedCorrs())

        elif ftr_name == "dcorrs":
            return get_unique_matrix_vals(sample.getDelayedCorrs(1))

        elif ftr_name == "tricorrs":
            return get_unique_3d_tensor_vals(get_3rd_order_corrs(sample.getStates()))

        elif ftr_name == "num-distr":
            # return sample.get_num_active_distr()
            raise NotImplementedError("TO-DO")

        else:
            raise ValueError(f"'{ftr_name}' is not a valid feature")

    @staticmethod
    def get_padded_interval(min_val, max_val, pad):
        interval = [
            min_val - (max_val - min_val) * pad,
            max_val + (max_val - min_val) * pad,
        ]
        return interval

    @staticmethod  # move to utils
    def _make_dir(dir_path):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Directory '{dir_path}' was created.")
            else:
                print(f"Directory '{dir_path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")

    @staticmethod
    def increment_path_num(path):
        pattern = r"(.*?)(\d+)(\.\w+)$"
        return re.sub(
            pattern, lambda m: f"{m.group(1)}{int(m.group(2))+1}{m.group(3)}", path
        )
