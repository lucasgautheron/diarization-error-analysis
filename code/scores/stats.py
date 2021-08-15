#!/usr/bin/env python3
# find ../../data/solomon-ana/output/clips -maxdepth 1 -type f -name "tsimane*" -exec sbatch -c 2 --mem 1G -o vtc ./apply.sh {} --device=cpu \;

# while read in; do sbatch -c 2 --mem 1G -o vtc ./apply.sh "../../data/solomon-ana/output/clips/$in" --device=cpu; done < ../../data/solomon-ana/output/clips/vtc.txt

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import datalad.api
from functools import partial
import librosa
import multiprocessing as mp
import numpy as np
from os.path import join as opj
from os.path import basename, exists, splitext

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from matplotlib import pyplot as plt

import pandas as pd
import pickle
from pyannote.core import Annotation, Segment, Timeline
import soundfile
from scipy.stats import poisson

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

def mean_scores(spkr, vocs, data, time_scale, range_onset, range_offset):
    s = []
    for voc in vocs:
        onset, offset = voc
        onset -= range_onset
        offset -= range_onset
        
        s.append(
            np.mean(data[spkr, int(onset*time_scale):int(offset*time_scale)])
        )
    return s

def process_clip(am, annotator, clip):
    index, ann = clip
    recording_filename, range_onset, range_offset, clip_path = index

    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    vtc_speakers = ['KCHI', 'CHI', 'FEM', 'MAL']

    data = None
    for i, speaker in enumerate(vtc_speakers):
        path = clip_path + f'/{speaker}.wav'

        if not exists(path):
            return None

        signal, sr = librosa.load(path, sr = None)

        time_scale = len(signal)/(range_offset-range_onset+4000)
        signal = signal[int(2000*time_scale):-int(2000*time_scale)]

        if data is not None:
            data = np.vstack((data, signal))
        else:
            data = signal

    segments = am.get_segments(ann)
    if not len(segments):
        return None
    segments = segments[segments['speaker_type'].isin(speakers)]

    vtc = {
        speaker: segments_to_annotation(segments[(segments['set'] == 'vtc') & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
        for speaker in speakers
    }
    
    truth = {
        speaker: segments_to_annotation(segments[(segments['set'] == annotator) & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
        for speaker in speakers
    }
    
    stats = {}
    for i, speaker_A in enumerate(speakers):
        vtc[f'{speaker_A}_vocs_explained'] = vtc[speaker_A].crop(truth[speaker_A], mode = 'loose')
        vtc[f'{speaker_A}_vocs_fp'] = extrude(vtc[speaker_A], vtc[f'{speaker_A}_vocs_explained'])
        vtc[f'{speaker_A}_vocs_fn'] = extrude(truth[speaker_A], truth[speaker_A].crop(vtc[speaker_A], mode = 'loose'))

        for speaker_B in speakers:
            vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = vtc[f'{speaker_A}_vocs_fp'].crop(truth[speaker_B], mode = 'loose')
            
            for speaker_C in speakers:
                if speaker_C != speaker_B and speaker_C != speaker_A:
                    vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = extrude(
                        vtc[f'{speaker_A}_vocs_fp_{speaker_B}'],
                        vtc[f'{speaker_A}_vocs_fp_{speaker_B}'].crop(truth[speaker_C], mode = 'loose')
                    )

        stats[f'{speaker_A}_scores'] = mean_scores(
            i, vtc[speaker_A], data, time_scale, range_onset, range_offset
        )
        stats[f'{speaker_A}_vocs_explained'] = mean_scores(
            i, vtc[f'{speaker_A}_vocs_explained'], data, time_scale, range_onset, range_offset
        )
        stats[f'{speaker_A}_truth'] = mean_scores(
            i, truth[speaker_A], data, time_scale, range_onset, range_offset
        )

        for j, speaker_B in enumerate(speakers):
            stats[f'{speaker_A}_vocs_fp_{speaker_B}_false'] = mean_scores(
                i, vtc[f'{speaker_A}_vocs_fp_{speaker_B}'], data, time_scale, range_onset, range_offset
            )
            stats[f'{speaker_A}_vocs_fp_{speaker_B}_true'] = mean_scores(
                j, vtc[f'{speaker_A}_vocs_fp_{speaker_B}'], data, time_scale, range_onset, range_offset
            )

    return stats
    

def generate_stats(parameters):
    annotator = parameters['annotator']

    project = ChildProject(parameters['path'])
    am = AnnotationManager(project)
    am.read()

    intersection = AnnotationManager.intersection(
        am.annotations, ['vtc', annotator]
    )

    intersection['clip_path'] = intersection.apply(
        lambda r: opj('output/scores/', splitext(basename(r['recording_filename']))[0] + '_' + str(r['range_onset']) + '_' + str(r['range_offset'])),
        axis = 1
    )

    with mp.Pool(processes = 48) as pool:
        stats = [
            process_clip(am, annotator, clip)
            for clip in intersection.groupby([
                'recording_filename', 'range_onset', 'range_offset', 'clip_path'
            ])
        ]

    return stats
    
if __name__ == '__main__':
    #annotators = pd.read_csv('input/annotators.csv')[:-3]
    annotators = pd.read_csv('input/annotators.csv')[:-3]
    annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
    stats = [generate_stats(annotator) for annotator in annotators.to_dict(orient = 'records')]
    stats = sum(stats, [])
    stats = [s for s in stats if s is not None]

    combined = {}

    for k in stats[0].keys():
        combined[k] = []
        for s in stats:
            if s is None:
                continue
            combined[k] += s[k]

    for k in combined.keys():
        if len(combined[k]) == 0:
            continue
        print(k,
            np.mean(combined[k]),
            np.quantile(combined[k], 0.1),
            np.quantile(combined[k], 0.9),
            len(combined[k])
        )

    with open('output/scores.pickle', 'wb') as fp:
        pickle.dump(combined, fp, pickle.HIGHEST_PROTOCOL)

    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    colors = ['red', 'orange', 'green', 'blue']

    fig, axes = plt.subplots(4, 4)
    for i, speaker_A in enumerate(speakers):
        for j, speaker_B in enumerate(speakers):
            ax = axes.flatten()[4*i+j]

            if i == j:
                data = combined[f'{speaker_B}_vocs_explained']
            else:
                data = combined[f'{speaker_B}_vocs_fp_{speaker_A}_false']

            hist, edges = np.histogram(data, bins = 20, range = (0,1))
            bins = (edges[:-1] + edges[1:]) / 2
            width = bins[1]-bins[0]

            low = poisson.ppf((1-0.68)/2, hist)
            high = poisson.ppf(1-(1-0.68)/2, hist)
            yerr = np.array([
                hist-low, high-hist
            ])
            yerr = np.nan_to_num(yerr)

            s = np.sum(hist)
            hist = hist/s
            yerr = yerr/s

            ax.errorbar(
                bins, hist,
                yerr = yerr,
                color = colors[j],
                ls='none',
                elinewidth=0.5
            )
            ax.step(
                bins, hist,
                color = colors[j],
                linewidth = 1,
                where = 'mid'
            )
            ax.axvline(np.nanmean(data), linestyle = '--', linewidth = 0.5, color = '#333', alpha = 1)

            ax.set_xlim(0,1)
            ax.set_ylim(0,0.5)

            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if i == 0:
                ax.xaxis.tick_top()
                ax.set_xticks([0.5])
                ax.set_xticklabels([speakers[j]])

            if i == 3:
                ax.set_xticks(np.linspace(0,1,4,endpoint=False))
                ax.set_xticklabels(np.linspace(0,1,4,endpoint=False))

            if j == 0:
                ax.set_yticks([0.25])
                ax.set_yticklabels([speakers[i]])
            
            if j == 3:
                ax.yaxis.tick_right()
                ax.set_yticks(np.linspace(0,0.5,4,endpoint=False))
                ax.set_yticklabels(['' for i in np.linspace(0,0.5,4,endpoint=False)])


    ax = fig.add_subplot(111, frameon=False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("vocalizations average scores ($\hat{y}$)", labelpad=20)

    fig.suptitle("VTC scores for true and false positive vocalizations")
    fig.subplots_adjust(wspace = 0, hspace = 0)

    fig.set_size_inches(set_size(450, 1))
    fig.savefig('output/scores.pdf')
    fig.savefig('output/scores.pgf')

    plt.close(fig)

    scores = np.zeros((len(speakers), len(stats)))
    vocs = np.zeros((len(speakers), len(stats)))
    truth = np.zeros((len(speakers), len(stats)))
    tp = np.zeros((len(speakers), len(stats)))
    r2 = np.zeros(len(speakers))
    baseline = np.zeros(len(speakers))

    for i, speaker in enumerate(speakers):
        for j, s in enumerate(stats):
            scores[i,j] = np.nansum(s[f'{speaker}_scores'])
            tp[i,j] = len(s[f'{speaker}_vocs_explained'])
            vocs[i,j] = len(s[f'{speaker}_scores'])
            truth[i,j] = len(s[f'{speaker}_truth'])

        r2[i] = np.corrcoef(scores[i,:],tp[i,:])[0,1]**2
        baseline[i] = np.corrcoef(vocs[i,:],tp[i,:])[0,1]**2

    print(r2)
    print(baseline)

    table = pd.DataFrame(np.array([r2, baseline]).T, columns = ['scores', 'vocs'])
    table = table.assign(speaker = speakers)

    table.to_latex(
        'output/scores_as_predictor.tex',
        columns = ['speaker', 'vocs', 'scores'],
        index = False,
        float_format = lambda x: f'{x:.2f}',
        column_format = '|c|c|c|',
        bold_rows = True,
        label = 'table:scores_as_predictor',
        caption = """
        Squared pearson correlation ($R^2$) with the amount of true positive.
        Scores alone are slightly better predictors than vocalization counts. 
        """
    )

