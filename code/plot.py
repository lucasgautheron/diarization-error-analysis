import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

fit = pd.read_csv('fit.csv')

fig = plt.figure(figsize=(8,8))
axes = [fig.add_subplot(4,4,i+1) for i in range(4*4)]

for i in range(4*4):
    ax = axes[i]
    row = i//4+1
    col = i%4+1
    label = f'confusion.{row}.{col}'

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(0,5)
    ax.set_xlim(0,1)

    if row == 4:
        ax.set_xticks(np.linspace(0,1,4, endpoint = False))
        ax.set_xticklabels(np.linspace(0,1,4, endpoint = False))

    ax.hist(fit[label], bins = np.linspace(0,1,40), density = True)

fig.subplots_adjust(wspace = 0, hspace = 0)

plt.show()