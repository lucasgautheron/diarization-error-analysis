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
import pandas as pd
import pickle
from pyannote.core import Annotation, Segment, Timeline

import stan

parser = argparse.ArgumentParser(description = 'model3')
parser.add_argument('--group', default = 'child', choices = ['corpus', 'child'])
parser.add_argument('--chains', default = 4, type = int)
parser.add_argument('--samples', default = 2000, type = int)
parser.add_argument('--validation', default = 0, type = float)
args = parser.parse_args()

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

        for speaker_A in speakers:
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


        d = {}
        for i, speaker_A in enumerate(speakers):
            for j, speaker_B in enumerate(speakers):
                if i != j:
                    z = len(vtc[f'{speaker_A}_vocs_fp_{speaker_B}'])
                else:
                    z = min(len(vtc[f'{speaker_A}_vocs_explained']), len(truth[speaker_A]))

                d[f'vtc_{i}_{j}'] = z

            d[f'truth_{i}'] = len(truth[speaker_A])
            d['child'] = child

        data.append(d)

    return pd.DataFrame(data).assign(
        corpus = corpus
    )

stan_code = """
data {
  int<lower=1> n_clips;   // number of clips
  int<lower=1> n_groups; // number of groups
  int<lower=1> n_classes; // number of classes
  int group[n_clips];
  int vtc[n_clips,n_classes,n_classes];
  int truth[n_clips,n_classes];
  int n_validation;
}

parameters {
  matrix<lower=0,upper=1>[n_classes,n_classes] mus;
  matrix<lower=1>[n_classes,n_classes] etas;
  matrix<lower=0,upper=1>[n_classes,n_classes] group_confusion[n_groups];
}

transformed parameters {
  matrix<lower=0>[n_classes,n_classes] alphas;
  matrix<lower=0>[n_classes,n_classes] betas;

  alphas = mus * etas;
  betas = (1-mus) * etas;
}

model {
    for (k in n_validation:n_clips) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                vtc[k,i,j] ~ binomial(truth[k,j], group_confusion[group[k],j,i]);
            }
        }
    }

    for (i in 1:n_classes) {
        for (j in 1:n_classes) {
            mus[i,j] ~ beta(1,1);
            etas[i,j] ~ pareto(1,1.5);
        }
    }

    for (c in 1:n_groups) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                group_confusion[c,i,j] ~ beta(alphas[i,j], betas[i,j]);
            }
        }
    }
}

generated quantities {
    int pred[n_clips,n_classes,n_classes];
    matrix[n_classes,n_classes] probs[n_groups];

    matrix[n_classes,n_classes] log_lik[n_clips];

    for (c in 1:n_groups) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                probs[c,i,j] = beta_rng(alphas[i,j], betas[i,j]);
            }
        }
    }

    for (k in 1:n_clips) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                pred[k,i,j] = binomial_rng(truth[k,j], probs[group[k],i,j]);
                log_lik[k,i,j] = beta_lpdf(probs[group[k],i,j] | alphas[i,j], betas[i,j]);
                log_lik[k,i,j] += binomial_lpmf(vtc[k,i,j] | truth[k,j], probs[group[k],i,j]);
            }
        }
    }
}
"""

if __name__ == "__main__":
    annotators = pd.read_csv('input/annotators.csv')
    annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))

    with mp.Pool(processes = 8) as pool:
        data = pd.concat(pool.map(compute_counts, annotators.to_dict(orient = 'records')))

    data = data.sample(frac = 1)

    vtc = np.moveaxis([[data[f'vtc_{j}_{i}'].values for i in range(4)] for j in range(4)], -1, 0)
    truth = np.transpose([data[f'truth_{i}'].values for i in range(4)])

    print(vtc.shape)

    data = {
        'n_clips': truth.shape[0],
        'n_classes': truth.shape[1],
        'n_groups': data[args.group].nunique(),
        'n_validation': int(truth.shape[0]*args.validation),
        'group': 1+data[args.group].astype('category').cat.codes.values,
        'truth': truth.astype(int),
        'vtc': vtc.astype(int)
    }

    print(f"clips: {data['n_clips']}")
    print(f"groups: {data['n_groups']}")
    print("true vocs: {}".format(np.sum(data['truth'])))
    print("vtc vocs: {}".format(np.sum(data['vtc'])))

    with open('data_model3.pickle', 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

    posterior = stan.build(stan_code, data = data)
    fit = posterior.sample(num_chains = args.chains, num_samples = args.samples)
    df = fit.to_frame()
    df.to_parquet('fit_model3.parquet')

