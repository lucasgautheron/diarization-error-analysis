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

parser = argparse.ArgumentParser(description = 'plot_sim_null')
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

fit = fit[fit['chi_adu_coef'] < 0.0001]
sim_beta_adu = np.zeros(len(fit))
sim_beta_och = np.zeros(len(fit))
sim_r_adu = np.zeros(len(fit))
sim_r_och = np.zeros(len(fit))
true = np.zeros(len(fit))

for i in range(len(fit)):
    f = fit.iloc[i]

    chi_vtc = np.array([f[f'sim_vtc.{k+1}.1'] for k in range(n_sim)])
    adu_vtc = np.array([f[f'sim_vtc.{k+1}.3']+f[f'sim_vtc.{k+1}.4'] for k in range(n_sim)])
    och_vtc = np.array([f[f'sim_vtc.{k+1}.2'] for k in range(n_sim)])

    regr = LinearRegression()
    regr.fit(adu_vtc.reshape(-1, 1), chi_vtc)
    sim_beta_adu[i] = regr.coef_[0]
    sim_r_adu[i] = np.corrcoef(adu_vtc, chi_vtc)[0,1]

    regr = LinearRegression()
    regr.fit(och_vtc.reshape(-1, 1), chi_vtc)
    sim_beta_och[i] = regr.coef_[0]
    sim_r_och[i] = np.corrcoef(och_vtc, chi_vtc)[0,1]


fig, ax = plt.subplots(1,2,figsize=set_size(450,1,0.5))
ax[0].hist(sim_beta_adu, histtype = 'step', bins = 20, density = True, label = '$\\mathrm{CHI} = K \\mathrm{ADU} + I$')
ax[0].hist(sim_beta_och, histtype = 'step', bins = 20, density = True, label = '$\\mathrm{CHI} = K \\mathrm{OCH} + I$')
ax[0].set_xlabel('$\\hat{K}$')
ax[0].set_ylabel('$p(\\hat{K}|K=0)$')
ax[0].legend(loc = 'upper left', bbox_to_anchor=[0, 1.25], frameon=False)
ax[0].set_xlim(-1,1)

ax[1].hist(sim_r_adu, histtype = 'step', bins = 20, density = True, label = '$R(\\mathrm{CHI},\\mathrm{ADU})$')
ax[1].hist(sim_r_och, histtype = 'step', bins = 20, density = True, label = '$R(\\mathrm{CHI},\\mathrm{OCH})$')
ax[1].set_xlabel('$R$')
ax[1].set_ylabel('$p(R|K=0)$')
ax[1].legend(loc = 'upper left', bbox_to_anchor=[0, 1.25], frameon=False)
ax[1].set_xlim(-1,1)

fig.savefig(args.output, bbox_inches = 'tight')


