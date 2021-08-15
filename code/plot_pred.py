#!/usr/bin/env python3

import pandas as pd
import pickle
import numpy as np

import argparse
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
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    if ratio is None:
        ratio = (5 ** 0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * ratio
    return fig_width_in, fig_height_in

parser = argparse.ArgumentParser(description = 'plot_pred')
parser.add_argument('data')
parser.add_argument('fit')
args = parser.parse_args()

with open(args.data, 'rb') as fp:
    data = pickle.load(fp)

fit = pd.read_parquet(args.fit)

fig = plt.figure(figsize=set_size(450, 1, 1))
axes = [fig.add_subplot(2,2,i+1) for i in range(4)]

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

n_values = data['n_validation']

for i in range(4):
    ax = axes[i]
    row = i//2+1
    col = i%2+1
    
    truth = data['truth'][:n_values,i]
    vtc = np.sum(data['vtc'][:n_values,i,:], axis = 1)
    pred = np.array([fit[f'pred.{k+1}.{i+1}'] for k in range(n_values)])
    errors = np.quantile(pred, [(1-0.68)/2, 1-(1-0.68)/2], axis = 1)
    pred = np.mean(pred, axis = 1)

    mask = (vtc > 0) & (pred > 0)

    ax.set_xlim(1,1000)
    ax.set_ylim(1,1000)

    ax.set_xscale('log')
    ax.set_yscale('log')

    slopes_x = np.logspace(0,3,num=3)
    ax.plot(slopes_x, slopes_x, color = '#ddd', lw = 0.5)
    ax.scatter(vtc[mask], pred[mask], s = 1)
    ax.errorbar(vtc[mask], pred[mask], [pred[mask]-errors[0,mask],errors[1,mask]-pred[mask]], ls='none', elinewidth = 1)
    ax.scatter(vtc[(vtc > 0) & (truth > 0)], truth[(vtc > 0) & (truth > 0)], s = 0.5, color = 'red')

    r2 = np.corrcoef(vtc, pred)[0,1]**2
    baseline = np.corrcoef(vtc, truth)[0,1]**2

    print(speakers[i], r2, baseline)

    ax.text(2, 400, f'{speakers[i]}\n$R^2={r2:.2f}$\nbaseline = {baseline:.2f}', ha = 'left', va = 'center')

    if col == 2:
        ax.yaxis.tick_right()

    ax.set_xticks([10**i for i in range(3)])
    ax.set_yticks([10**i for i in range(3)])

    ax.set_xticklabels([f'$10^{i}$' for i in range(3)])
    ax.set_yticklabels([f'$10^{i}$' for i in range(3)])



#fig.suptitle("$\mu_{eff}$ distribution")
fig.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig('pred.pdf')
plt.show()

