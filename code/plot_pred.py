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
parser.add_argument('output')
args = parser.parse_args()

with open(args.data, 'rb') as fp:
    data = pickle.load(fp)

fit = pd.read_parquet(args.fit)

fig = plt.figure(figsize=set_size(450, 1, 1))
axes = [fig.add_subplot(4,4,i+1) for i in range(4*4)]

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

n_values = data['n_validation']

for i in range(4*4):
    ax = axes[i]
    row = i//4+1
    col = i%4+1
    
    truth = data['truth'][:n_values,row-1]
    #vtc = np.sum(data['vtc'][:n_values,i,:], axis = 1)
    vtc = np.array(data['vtc'][:n_values,col-1,row-1])
    pred_dist = np.array([fit[f'pred.{k+1}.{col}.{row}'] for k in range(n_values)])
    errors = np.quantile(pred_dist, [(1-0.68)/2, 1-(1-0.68)/2], axis = 1)
    pred = np.mean(pred_dist, axis = 1)

    # p = np.zeros(n_values)
    # for k in range(n_values):
    #     dy = np.abs(pred[k]-vtc[k])
    #     more_extreme = pred_dist[k,np.abs(pred_dist[k,:]-pred[k])>dy]
    #     p[k] = len(more_extreme)/pred_dist.shape[1]

    # chi_squared = -2*np.nansum(np.ma.log(p))/n_values

    # print(p.shape)
    # print(p)
    # print(chi_squared)

    # log_lik = np.array([fit[f'log_lik.{k+1}.{i+1}.{i+1}'] for k in range(n_values)])
    # print(log_lik)
    # log_lik = np.mean(log_lik)
    # print(log_lik)
    # print(np.exp(log_lik))

    mask = (vtc > 0) & (pred > 0)

    ax.set_xlim(1,1000)
    ax.set_ylim(1,1000)

    ax.set_xscale('log')
    ax.set_yscale('log')

    slopes_x = np.logspace(0,3,num=3)
    ax.plot(slopes_x, slopes_x, color = '#ddd', lw = 0.5)
    ax.scatter(vtc[mask], pred[mask], s = 1)
    ax.errorbar(vtc[mask], pred[mask], [pred[mask]-errors[0,mask],errors[1,mask]-pred[mask]], ls='none', elinewidth = 0.5)
    ax.scatter(vtc[(vtc > 0) & (truth > 0)], truth[(vtc > 0) & (truth > 0)], s = 0.5, color = 'red')

    #r2 = np.corrcoef(vtc, pred)[0,1]**2
    #baseline = np.corrcoef(vtc, truth)[0,1]**2
    #print(speakers[i], r2, baseline)
    #ax.text(2, 400, speakers[i], ha = 'left', va = 'center')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if col == 4:
        ax.yaxis.tick_right()

    if row == 1:
        ax.xaxis.tick_top()
        ax.set_xticks([10**1.5])
        ax.set_xticklabels([speakers[col-1]])

    if row == 4:
        ax.set_xticks(np.power(10, np.arange(1,4)))
        ax.set_xticklabels([f'10$^{i}$' for i in [1,2,3]])

    if col == 1:
        ax.set_yticks([10**1.5])
        ax.set_yticklabels([speakers[row-1]])
    
    if col == 4:
        ax.yaxis.tick_right()
        ax.set_yticks(np.power(10, np.arange(1,4)))
        ax.set_yticklabels([f'10$^{i}$' for i in [1,2,3]])

plt.xlabel('')

#fig.suptitle("$\mu_{eff}$ distribution")
fig.subplots_adjust(wspace = 0, hspace = 0)
plt.savefig(args.output)
plt.show()


