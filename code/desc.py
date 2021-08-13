#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import argparse

import datalad.api
from os.path import join as opj
from os.path import basename, exists

import multiprocessing as mp

import numpy as np
from scipy.stats import binom
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description = 'model3')
parser.add_argument('--group', default = 'child', choices = ['corpus', 'child'])
parser.add_argument('--chains', default = 4, type = int)
parser.add_argument('--samples', default = 2000, type = int)
args = parser.parse_args()

def set_size(width, ratio):
    return width/72.27, ratio*width/72.27

def extrude(self, removed, mode: str = 'intersection'):
    if isinstance(removed, Segment):
        removed = Timeline([removed])

    truncating_support = removed.gaps(support=self.extent())
    # loose for truncate means strict for crop and vice-versa
    if mode == "loose":
        mode = "strict"
    elif mode == "strict":
        mode = "loose"
    
    return self.crop(truncating_support, mode=mode)

def compute_counts(parameters):
    corpus = parameters['corpus']
    annotator = parameters['annotator']
    speakers = ['CHI', 'OCH', 'FEM', 'MAL']

    project = ChildProject(parameters['path'])
    am = AnnotationManager(project)
    am.read()

    intersection = AnnotationManager.intersection(
        am.annotations, ['vtc', annotator]
    )
    intersection['onset'] = intersection.apply(lambda r: np.arange(r['range_onset'], r['range_offset'], 15000), axis = 1)
    intersection = intersection.explode('onset')
    intersection['range_onset'] = intersection['onset']
    intersection['range_offset'] = (intersection['range_onset']+15000).clip(upper = intersection['range_offset'])

    intersection['path'] = intersection.apply(
        lambda r: opj(project.path, 'annotations', r['set'], 'converted', r['annotation_filename']),
        axis = 1
    )
    datalad.api.get(list(intersection['path'].unique()))

    intersection = intersection.merge(project.recordings[['recording_filename', 'child_id']], how = 'left')
    intersection['child'] = corpus + '_' + intersection['child_id'].astype(str)

    data = []
    for child, ann in intersection.groupby('child'):
        print(corpus, child)

        segments = am.get_collapsed_segments(ann)
        if 'speaker_type' not in segments.columns:
            continue

        segments = segments[segments['speaker_type'].isin(speakers)]
        
        vtc = {
            speaker: segments_to_annotation(segments[(segments['set'] == 'vtc') & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
            for speaker in speakers
        }

        truth = {
            speaker: segments_to_annotation(segments[(segments['set'] == annotator) & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
            for speaker in speakers
        }

        for i, speaker_A in enumerate(speakers):
            vtc[f'{speaker_A}_vocs_explained'] = vtc[speaker_A].crop(truth[speaker_A], mode = 'loose')
            vtc[f'{speaker_A}_vocs_fp'] = extrude(vtc[speaker_A], vtc[f'{speaker_A}_vocs_explained'])
            vtc[f'{speaker_A}_vocs_fn'] = extrude(truth[speaker_A], truth[speaker_A].crop(vtc[speaker_A], mode = 'loose'))
            vtc[f'{speaker_A}_vocs_unexplained'] = extrude(vtc[speaker_A], vtc[f'{speaker_A}_vocs_explained'])

            for speaker_B in speakers:
                vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = vtc[f'{speaker_A}_vocs_fp'].crop(truth[speaker_B], mode = 'loose')
                vtc[f'{speaker_A}_vocs_unexplained'] = extrude(vtc[f'{speaker_A}_vocs_unexplained'], vtc[f'{speaker_A}_vocs_unexplained'].crop(truth[speaker_B], mode = 'loose'))
                
                for speaker_C in speakers:
                    if speaker_C != speaker_B and speaker_C != speaker_A:
                        vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = extrude(vtc[f'{speaker_A}_vocs_fp_{speaker_B}'], truth[speaker_C])
                

        d = {'child': child}
        for i, speaker_A in enumerate(speakers):
            for j, speaker_B in enumerate(speakers):
                if i != j:
                    z = len(vtc[f'{speaker_A}_vocs_fp_{speaker_B}'])
                else:
                    z = len(vtc[f'{speaker_A}_vocs_explained'])

                d[f'vtc_{speaker_A}_{speaker_B}'] = z

            if len(vtc[f'{speaker_A}_vocs_explained']) > len(truth[speaker_A]):
                print(speaker_A, child)

            d[f'truth_{speaker_A}'] = len(truth[speaker_A])
            d[f'unexplained_{speaker_B}'] = len(vtc[f'{speaker_A}_vocs_unexplained'])
         
        data.append(d)

    return pd.DataFrame(data).assign(
        corpus = corpus
    )

if __name__ == "__main__":
    annotators = pd.read_csv('input/annotators.csv')
    annotators = annotators[~annotators['annotator'].str.startswith('eaf_2021')]

    annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))

    with mp.Pool(processes = 8) as pool:
        data = pd.concat(pool.map(compute_counts, annotators.to_dict(orient = 'records')))

    data.to_csv('output/summary.csv', index = False)

    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    colors = ['red', 'orange', 'green', 'blue']

    fig, axes = plt.subplots(4, 4, figsize = (6,6))
    for i, speaker_A in enumerate(speakers):
        for j, speaker_B in enumerate(speakers):
            ax = axes.flatten()[4*i+j]

            x = data[f'truth_{speaker_A}'].values
            y = data[f'vtc_{speaker_B}_{speaker_A}'].values

            mask = (x > 0) & (y > 0)
            x = x[mask]
            y = y[mask]

            low = binom.ppf((1-0.68)/2, x, y/x)
            high = binom.ppf(1-(1-0.68)/2, x, y/x)

            mask = (~np.isnan(low)&(~np.isnan(high)))

            yerr = np.array([
                y[mask]-low[mask], high[mask]-y[mask]
            ])

            slopes_x = np.logspace(0,3,num=3)

            ax.plot(slopes_x, slopes_x, color = '#ddd', lw = 0.5)
            ax.plot(slopes_x, 0.1*slopes_x, color = '#ddd', lw = 0.5, linestyle = '--')
            ax.plot(slopes_x, 0.01*slopes_x, color = '#ddd', lw = 0.5, linestyle = '-.')

            ax.errorbar(
                x[mask], y[mask],
                yerr = yerr,
                color = colors[j],
                ls='none',
                elinewidth=0.5
            )
            ax.scatter(
                x, y,
                s = 0.75,
                color = colors[j]
            )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1,1000)
            ax.set_ylim(1,1000)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if i == 0:
                ax.xaxis.tick_top()
                ax.set_xticks([10**1.5])
                ax.set_xticklabels([speakers[j]])

            if i == 3:
                ax.set_xticks(np.power(10, np.arange(1,4)))
                ax.set_xticklabels([f'10$^{i}$' for i in [1,2,3]])

            if j == 0:
                ax.set_yticks([10**1.5])
                ax.set_yticklabels([speakers[i]])
            
            if j == 3:
                ax.yaxis.tick_right()
                ax.set_yticks(np.power(10, np.arange(1,4)))
                ax.set_yticklabels([f'10$^{i}$' for i in [1,2,3]])

    fig.subplots_adjust(wspace = 0, hspace = 0)

    fig.set_size_inches(set_size(450, 1))
    fig.savefig('output/summary.pdf')
    fig.savefig('output/summary.pgf')




