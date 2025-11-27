import numpy as np
import matplotlib.pyplot as plt


def corrcoeff(x, y):
    return np.corrcoef(x, y)[0, 1]


def plot_regression_line(ax, x, y, color='k', ls='-', lw=2):
    a, b = np.polyfit(x, y, 1)
    x0, x1 = min(x), max(x)
    y0, y1 = a * x0 + b, a * x1 + b
    ax.plot([x0, x1], [y0, y1], color=color, ls=ls, lw=lw)


def descriptive(df):

    dd  = df['Death Description']
    mag = df['Mag']
    fd  = df['Focal Depth']
    lat = df['Lat']

    fig, axs = plt.subplots(2, 2, figsize=(10,7), tight_layout=True)
    ivs      = [mag, fd, lat]
    colors   = ['b', 'r', 'g']

    for ax, x, c in zip(axs.ravel()[:3], ivs, colors):
        ax.scatter(x, dd, alpha=0.5, color=c)
        plot_regression_line(ax, x, dd, color='k', ls='-', lw=2)
        r = corrcoeff(x, dd)
        ax.text(
            0.70, 0.25, f'r = {r:.3f}', color=c,
            transform=ax.transAxes,
            bbox=dict(color='0.8', alpha=0.7)
        )

    xlabels = ['Magnitude', 'Focal Depth (×0.01)', 'Latitude']
    for ax, label in zip(axs.ravel()[:3], xlabels):
        ax.set_xlabel(label)

    axs[0,0].set_ylabel("Death Description")
    axs[1,0].set_ylabel("Death Description")

    for ax in axs[:,1]:
        ax.set_yticklabels([])

    ax = axs[1,1]

    i_low  = dd <= 2
    i_high = dd >= 3

    groups   = [i_low, i_high]
    g_colors = ['m', 'c']
    g_labels = ['Low-mortality', 'High-mortality']
    q_groups = [[1,2], [3,4]]
    ylocs    = [0.25, 0.75]

    for idx, color, label, qs, yloc in zip(groups, g_colors, g_labels, q_groups, ylocs):
        ax.scatter(mag[idx], dd[idx], alpha=0.5, color=color, label=label)
        plot_regression_line(ax, mag[idx], dd[idx], color=color, ls='-', lw=2)
        for q in qs:
            ax.plot(
                mag[idx].mean(), q,
                'o', color=color, mfc='white', ms=8
            )
        r = corrcoeff(mag[idx], dd[idx])
        ax.text(
            0.70, yloc, f'r = {r:.3f}', color=color,
            transform=ax.transAxes,
            bbox=dict(color='0.8', alpha=0.7)
        )

    ax.set_xlabel("Magnitude")
    ax.legend()

    panel_labels = ['a', 'b', 'c', 'd']
    for ax, s in zip(axs.ravel(), panel_labels):
        ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes)

    plt.show()
