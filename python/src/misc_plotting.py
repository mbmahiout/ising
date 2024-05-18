from src.isingfitter import IsingFitter
import src.utils as utils

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import ipywidgets as widgets


def get_hex_colors(num_colors, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, num_colors + 1))

    hex_colors = [
        "#%02x%02x%02x" % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        for rgb in colors[1:]
    ]
    return hex_colors


def plot_generalized(layout_spec, path=None):
    def save_figure(b):
        fig.write_image(path)
        print(f"Saved to {path}")

    num_rows = max([r for (r, _) in layout_spec.keys()])
    num_cols = max([c for (_, c) in layout_spec.keys()])

    fig = make_subplots(
        rows=num_rows, cols=num_cols, horizontal_spacing=0.1, vertical_spacing=0.1
    )

    curve_color = "rgba(250, 150, 0, 1)"
    fill_color = "rgba(250, 150, 0, 0.5)"

    for (r, c), specs in layout_spec.items():

        data = specs["data"]
        label = specs["label"]

        step_label = specs.get("step_label", "Steps")

        if isinstance(data[0], float):
            steps = specs.get("steps", list(range(len(data))))
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=data,
                    mode="lines",
                    name=f"{label}",
                    line=dict(color=curve_color),
                ),
                row=r,
                col=c,
            )

        else:
            data = [a.flatten() for a in data]

            av_data = np.mean(data, axis=1)
            min_data = np.min(data, axis=1)
            max_data = np.max(data, axis=1)
            steps = specs.get("steps", list(range(len(av_data))))

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=av_data,
                    mode="lines",
                    name=f"Av. {label}",
                    line=dict(color=curve_color),
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=min_data,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                ),
                row=r,
                col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=max_data,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=fill_color,
                    line=dict(width=0),
                    hoverinfo="skip",
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(title_text=step_label, title_font=dict(size=20), row=r, col=c)
        fig.update_yaxes(title_text=label, title_font=dict(size=20), row=r, col=c)

    fig.update_layout(
        height=400 * num_rows,
        width=400 * num_cols,
        showlegend=False,
        font=dict(size=14),
    )

    fig.show()

    if path is not None:  # to-do: also check if is notebook
        button = widgets.Button(description="Save Figure")
        button.on_click(save_figure)
        display(button)


###########################################################################################################


def convergence_plot(fitter: IsingFitter, plot_llh: bool = False):
    layout_spec = {
        (1, 1): {
            "data": fitter.fields_grads,
            "label": r"$\Large \nabla h$",
            "step_label": "",
        },
        (2, 1): {
            "data": fitter.couplings_grads,
            "label": r"$\Large \nabla J$",
            "step_label": "Step",
        },
        (1, 2): {
            "data": fitter.fields_history,
            "label": r"$\Large h$",
            "step_label": "",
        },
        (2, 2): {
            "data": fitter.couplings_history,
            "label": r"$\Large J$",
            "step_label": "Step",
        },
        (1, 3): {
            "data": fitter.sd_fields,
            "label": r"$\Large \sigma_h$",
            "step_label": "",
        },
        (2, 3): {
            "data": fitter.sd_couplings,
            "label": r"$\Large \sigma_J$",
            "step_label": "Step",
        },
    }

    if plot_llh:
        layout_spec[(3, 2)] = {"data": fitter.llhs, "label": r"$LLHs$"}

    plot_generalized(layout_spec)


###########################################################################################################


def plot_histogram(
    fig: go.FigureWidget,
    labels: list,
    colors: list,
    obs_datas: list,
    row: int,
    col: int,
    num_bins: int,
):

    all_data = np.concatenate(obs_datas)
    min_val, max_val = np.min(all_data), np.max(all_data)
    bin_width = (max_val - min_val) / num_bins
    bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    showlegend = row == 1 and col == 1
    for obs_data, label, color in zip(obs_datas, labels, colors):
        counts, _ = np.histogram(obs_data, bins=bin_edges, density=True)

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


def plot_mean_histograms(fig, labels, colors, means, num_bins):
    for i in range(len(means)):
        plot_histogram(
            fig, labels, colors, means[0], row=1, col=i + 1, num_bins=num_bins
        )


def plot_pcorr_histograms(fig, labels, colors, pcorrs, num_bins):
    for i in range(len(pcorrs)):
        plot_histogram(
            fig, labels, colors, pcorrs[0], row=2, col=i + 1, num_bins=num_bins
        )


def add_hist_annotations(fig):
    fig.add_annotation(
        text=r"$$\Large\text{Pair-wise covariance, } \langle \sigma_i \sigma_j \rangle$$",
        xref="paper",  # 'paper' refers to the entire figure from 0 to 1
        yref="paper",
        x=0.5,
        y=-0.1,
        showarrow=False,
        font=dict(
            size=22,
        ),
        align="center",
    )

    fig.add_annotation(
        text=r"$$\Large\text{Mean, } \langle \sigma_i \rangle$$",
        xref="paper",  # 'paper' refers to the entire figure from 0 to 1
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=22),
        align="center",
    )

    # vertical subtitle on the left
    fig.add_annotation(
        text=r"$$\Large\text{Relative frequency}$$",
        xref="paper",
        yref="paper",
        x=-0.075,
        y=0.5,
        showarrow=False,
        font=dict(size=22),
        textangle=-90,  # rotated 90 degrees
    )


def plot_empirical_histograms(
    all_samples, labels, num_cols=4, num_rows=2, num_bins=30, path=None
):
    def save_figure(b):
        fig.write_image(path)
        print(f"Saved to {path}")

    colors = get_hex_colors(len(all_samples[0]), "winter")

    means = utils.get_all_recording_means(all_samples)
    pcorrs = utils.get_all_recording_pcorrs(all_samples)

    fig = go.FigureWidget(
        make_subplots(
            rows=num_rows,
            cols=num_cols,
        )
    )

    plot_mean_histograms(fig, labels, colors, means, num_bins)
    plot_pcorr_histograms(fig, labels, colors, pcorrs, num_bins)

    add_hist_annotations(fig)

    fig.update_layout(
        height=400 * num_rows, width=400 * num_cols, margin=dict(l=100, t=40, b=70)
    )

    display(fig)

    if path is not None:  # to-do: also check if is notebook
        button = widgets.Button(description="Save Figure")
        button.on_click(save_figure)
        display(button)


###########################################################################################################


def add_scatter_annotations(fig):
    fig.update_yaxes(
        title_text=r"$$\Large \langle \sigma_i \rangle ^ {\text{Simulated}}$$",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$$\Large \langle \sigma_i \sigma_j \rangle ^ {\text{Simulated}}$$",
        row=2,
        col=1,
    )

    fig.add_annotation(
        text=r"$$\Large \langle \sigma_i \sigma_j \rangle ^ {\text{Analytic}}$$",
        xref="paper",  # 'paper' refers to the entire figure from 0 to 1
        yref="paper",
        x=0.5,
        y=-0.1,
        showarrow=False,
        font=dict(
            size=22,
        ),
        align="center",
    )

    fig.add_annotation(
        text=r"$$\Large \langle \sigma_i \rangle ^ {\text{Analytic}}$$",
        xref="paper",  # 'paper' refers to the entire figure from 0 to 1
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=22),
        align="center",
    )


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


def get_padded_interval(min_val, max_val, pad):
    interval = [
        min_val - (max_val - min_val) * pad,
        max_val + (max_val - min_val) * pad,
    ]
    return interval


def _add_comparison_scatter(fig, x_data, y_data, color, row, col):
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(color=color, size=7.5, symbol="circle"),
            opacity=0.75,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def plot_scatter_comparison(fig, color, obs_sim, obs_anal, row, col, pad=0.5):

    _add_comparison_scatter(fig, obs_anal, obs_sim, color, row, col)

    min_val, max_val = np.min(obs_anal), np.max(obs_anal)
    interval = get_padded_interval(min_val, max_val, pad)
    _add_id_line(fig, interval, row, col)


def plot_analytic_simulated_scatter_comparisons(
    all_obs_anal: list, all_obs_sim: list, color: str, path=None
):
    def save_figure(b):
        fig.write_image(path)
        print(f"Saved to {path}")

    num_rows = 2
    num_cols = len(all_obs_anal)

    fig = go.FigureWidget(
        make_subplots(
            rows=num_rows,
            cols=num_cols,
        )
    )

    for obs_anal, obs_sim, col in zip(
        all_obs_anal, all_obs_sim, range(1, num_cols + 1)
    ):
        plot_scatter_comparison(fig, color, obs_sim["m"], obs_anal["m"], row=1, col=col)
        plot_scatter_comparison(
            fig,
            color,
            obs_sim["chi"].flatten(),
            obs_anal["chi"].flatten(),
            row=2,
            col=col,
        )

    fig.update_layout(
        height=400 * num_rows, width=400 * num_cols, margin=dict(l=100, t=40, b=70)
    )

    fig.update_yaxes(
        title_text=r"$$\Large \langle \sigma_i \rangle ^ {\text{Simulated}}$$",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$$\Large \langle \sigma_i \sigma_j \rangle ^ {\text{Simulated}}$$",
        row=2,
        col=1,
    )

    add_scatter_annotations(fig)

    display(fig)

    if path is not None:
        button = widgets.Button(description="Save Figure")
        button.on_click(save_figure)
        display(button)
