import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets


def scatter_compare_2d_obs(gt: np.ndarray, est: np.ndarray):
    num_units = gt.shape[0]
    gt2plt = []
    est2plt = []
    for i in range(num_units):
        for j in [k for k in range(num_units) if k != i]:
            gt2plt.append(gt[i, j])
            est2plt.append(est[i, j])

    max_val = max(max(gt2plt), max(est2plt))
    min_val = min(min(gt2plt), min(est2plt))
    interval = np.linspace(min_val * 2, max_val * 2, 100)

    plt.scatter(gt2plt, est2plt, color="dodgerblue", alpha=0.75)
    plt.plot(interval, interval, linestyle="--", color="black")

    plt.xlim([min_val * 1.2, max_val * 1.2])
    plt.ylim([min_val * 1.2, max_val * 1.2])
    plt.show()
    plt.close()


def scatter_compare_1d_obs(gt: np.ndarray, est: np.ndarray):
    max_val = max(max(gt), max(est))
    min_val = min(min(gt), min(est))
    interval = np.linspace(min_val * 2, max_val * 2, 100)

    plt.scatter(gt, est, color="dodgerblue", alpha=0.75)
    plt.plot(interval, interval, linestyle="--", color="black")

    plt.xlim([min_val * 1.2, max_val * 1.2])
    plt.ylim([min_val * 1.2, max_val * 1.2])
    plt.show()
    plt.close()


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
