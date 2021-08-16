#!/usr/bin/env python3

import pandas as pd
import pickle
import numpy as np

import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    "font.serif" : "Times New Roman",
    'text.usetex': True,
    'pgf.rcfonts': False,
})

from sklearn.linear_model import LinearRegression

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

speakers = ['CHI', 'OCH', 'FEM', 'MAL']

#n_simulations = fit['n_sim'].iloc[0]
n_sim = data['n_sim']

fit = fit[-1000:]
sim = np.zeros(len(fit))
true = np.zeros(len(fit))

for i in range(len(fit)):
    f = fit.iloc[i]
    true_beta = f['chi_adu_coef']

    chi_truth = np.array([f[f'sim_truth.{k+1}.1'] for k in range(n_sim)])
    adu_truth = np.array([f[f'sim_truth.{k+1}.3']+f[f'sim_truth.{k+1}.4'] for k in range(n_sim)])
    
    chi_vtc = np.array([f[f'sim_vtc.{k+1}.1'] for k in range(n_sim)])
    adu_vtc = np.array([f[f'sim_vtc.{k+1}.3']+f[f'sim_vtc.{k+1}.4'] for k in range(n_sim)])

    regr = LinearRegression()
    regr.fit(adu_vtc.reshape(-1, 1), chi_vtc)

    sim[i] = regr.coef_[0]
    true[i] = true_beta

fig, ax = plt.subplots(1,1,figsize=set_size(450,1,1))
ax.scatter(true, sim, s = 1)
ax.plot(np.linspace(0,5,4), np.linspace(0,5,4), color = 'black')
ax.set_xlabel('true $\\beta$')
ax.set_ylabel('fit $\\hat{\\beta}$')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
fig.savefig(args.output)


