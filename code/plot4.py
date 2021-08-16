#!/usr/bin/env python3

import argparse

import pandas as pd
import pickle
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

import seaborn as sns

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


parser = argparse.ArgumentParser(description = 'plot_pred')
parser.add_argument('data')
parser.add_argument('fit')
parser.add_argument('output')
args = parser.parse_args()

with open(args.data, 'rb') as fp:
    data = pickle.load(fp)

fit = pd.read_parquet(args.fit)

fig = plt.figure(figsize=set_size(450, 1, 1))
axes = [fig.add_subplot(4,4,i+1) for i in range(4*4)]

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

n_groups = data['n_groups']

for i in range(4*4):
    ax = axes[i]
    row = i//4+1
    col = i%4+1
    label = f'{row}.{col}'

    #mus = np.hstack([fit[f'alphas.{k}.{label}']/(fit[f'alphas.{k}.{label}']+fit[f'betas.{k}.{label}']).values for k in range(1,n_groups+1)])
    #etas = np.hstack([(fit[f'alphas.{k}.{label}']+fit[f'betas.{k}.{label}']).values for k in range(1,n_groups+1)])
    #etas = np.log10(etas)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(0,3)
    ax.set_xlim(0,1)

    if row == 1:
        ax.xaxis.tick_top()
        ax.set_xticks([0.5])
        ax.set_xticklabels([speakers[col-1]])

    if row == 4:
        ax.set_xticks(np.linspace(0.25,1,3, endpoint = False))
        ax.set_xticklabels(np.linspace(0.25,1,3, endpoint = False))

    if col == 1:
        ax.set_yticks([1.5])
        ax.set_yticklabels([speakers[row-1]])
    
    if col == 4:
        ax.yaxis.tick_right()
        ax.set_yticks(np.arange(1,3))
        ax.set_yticklabels([f'10$^{i}' for i in np.arange(1,3)])

    kplt = sns.kdeplot(fit[f'mus.{label}'], fit[f'etas.{label}'].apply(np.log), shade=True, cmap="viridis", ax = ax)
    #kplt = sns.kdeplot(mus, etas, shade=True, cmap="viridis", ax = ax)
    kplt.set(xlabel = None, ylabel = None)
    ax.axvline(np.mean(fit[f'mus.{label}']), linestyle = '--', linewidth = 0.5, color = '#333', alpha = 1)

fig.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig(args.output)
plt.show()

