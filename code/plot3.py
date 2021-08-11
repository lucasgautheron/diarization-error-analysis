#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    "font.serif" : "Times New Roman",
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def set_size(width, fraction=1, ratio = None):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if ratio is None:
        ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    return fig_width_in, fig_height_in


fit = pd.read_parquet('fit.parquet')

fig = plt.figure(figsize=set_size(450, 1, 1))
axes = [fig.add_subplot(4,4,i+1) for i in range(4*4)]

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

for i in range(4*4):
    ax = axes[i]
    row = i//4+1
    col = i%4+1
    label = f'{row}.{col}'

    data = np.hstack([fit[f'alphas.{k}.{label}']/(fit[f'alphas.{k}.{label}']+fit[f'betas.{k}.{label}']).values for k in range(1,59)])
    #data = np.hstack([(fit[f'group_mus.{k}.{label}']).values for k in range(1,59)])
    #data = fit[f'mus.{label}'].values    
  
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0,5)
    ax.set_xlim(0,1)

    low = np.quantile(data, 0.0275)
    high = np.quantile(data, 0.975)

    if row == 1:
        ax.xaxis.tick_top()
        ax.set_xticks([0.5])
        ax.set_xticklabels([speakers[col-1]])

    if row == 4:
        ax.set_xticks(np.linspace(0.25,1,3, endpoint = False))
        ax.set_xticklabels(np.linspace(0.25,1,3, endpoint = False))

    if col == 1:
        ax.set_yticks([2.5])
        ax.set_yticklabels([speakers[row-1]])

    ax.hist(data, bins = np.linspace(0,1,40), density = True)
    ax.axvline(np.mean(data), linestyle = '--', linewidth = 0.5, color = '#333', alpha = 1)
    ax.text(0.5, 4.5, f'{low:.2f} - {high:.2f}', ha = 'center', va = 'center')

fig.suptitle("$\mu_{eff}$ distribution")
fig.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig('mu_eff.pdf')
plt.show()
