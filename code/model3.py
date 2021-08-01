#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import datalad.api
from os.path import join as opj
from os.path import basename, exists
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline

import stan
import numpy as np
from scipy.stats import beta

from matplotlib import pyplot as plt
import seaborn as sns

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

    segments = am.get_collapsed_segments(intersection)
    segments = segments.merge(project.recordings[['recording_filename', 'child_id']], how = 'left')
    segments['child'] = corpus + '_' + segments['child_id'].astype(str)

    segments = segments[segments['speaker_type'].isin(speakers)]

    data = []
    for child, _segments in segments.groupby('child'):
        vtc = {
            speaker: segments_to_annotation(_segments[(_segments['set'] == 'vtc') & (_segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
            for speaker in speakers
        }

        truth = {
            speaker: segments_to_annotation(_segments[(_segments['set'] == annotator) & (_segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
            for speaker in speakers
        }

        for speaker_A in speakers:
            vtc[f'{speaker_A}_vocs_explained'] = vtc[speaker_A].crop(truth[speaker_A], mode = 'loose')
            vtc[f'{speaker_A}_vocs_fp'] = extrude(vtc[speaker_A], vtc[f'{speaker_A}_vocs_explained'])
            vtc[f'{speaker_A}_vocs_fn'] = extrude(truth[speaker_A], truth[speaker_A].crop(vtc[speaker_A], mode = 'loose'))

            for speaker_B in speakers:
                vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = vtc[f'{speaker_A}_vocs_fp'].crop(truth[speaker_B], mode = 'loose')

        d = {}
        for i, speaker_A in enumerate(speakers):
            for j, speaker_B in enumerate(speakers):
                if i != j:
                    z = len(vtc[f'{speaker_A}_vocs_fp_{speaker_B}'])
                else:
                    z = len(truth[speaker_A]) - len(vtc[f'{speaker_A}_vocs_fn'])

                d[f'vtc_{i}_{j}'] = z

            d[f'truth_{i}'] = len(truth[speaker_A])
            d['child'] = child

        data.append(d)

    return pd.DataFrame(data).assign(
        corpus = corpus
    )

annotators = pd.read_csv('input/annotators.csv')
annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
data = pd.concat([compute_counts(annotator) for annotator in annotators.to_dict(orient = 'records')])

print(data)

vtc = np.moveaxis([[data[f'vtc_{i}_{j}'].values for i in range(4)] for j in range(4)], -1, 0)
truth = np.transpose([data[f'truth_{i}'].values for i in range(4)])

print(vtc.shape)

# random data
# n_clips = 200
# n_classes = 4
# expectation = np.array([15,1,5,1])

# confusion = np.zeros((n_classes, n_classes))
# for i in range(n_classes):
#     for j in range(n_classes):
#         confusion[i,j] = 0.9 if i == j else 0.05

# truth = np.random.poisson(expectation, size = (n_clips, n_classes))
# vtc = np.zeros((n_clips, n_classes))
# for k in range(n_clips):
#     for i in range(n_classes):
#         vtc[k,i] = np.sum(np.random.binomial(truth[k,:], confusion[:,i]))

data = {
    'n_clips': truth.shape[0],
    'n_classes': truth.shape[1],
    'n_children': data['child'].nunique(),
    'child': 1+data['child'].astype('category').cat.codes.values,
    'truth': truth.astype(int),
    'vtc': vtc.astype(int)
}

print(f"clips: {data['n_clips']}")
print("true vocs: {}".format(np.sum(data['truth'])))
print("vtc vocs: {}".format(np.sum(data['vtc'])))

plt.show()

stan_code = """
data {
  int<lower=1> n_clips;   // number of clips
  int<lower=1> n_children; // number of children
  int<lower=1> n_classes; // number of classes
  int child[n_clips];
  int vtc[n_clips,n_classes,n_classes];
  int truth[n_clips,n_classes];
}

parameters {
  matrix<lower=0,upper=1>[n_classes,n_classes] mus;
  matrix<lower=0>[n_classes,n_classes] logetas;
  matrix<lower=0,upper=1>[n_classes,n_classes] child_confusion[n_children];
}

transformed parameters {
  matrix<lower=1>[n_classes,n_classes] alphas;
  matrix<lower=1>[n_classes,n_classes] betas;

  alphas = mus * exp(logetas);
  betas = (1-mus) * exp(logetas);
}
model {
    for (k in 1:n_clips) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                vtc[k,i,j] ~ binomial(truth[k,i], child_confusion[child[k],i,j]);
            }
        }
    }

    for (i in 1:n_classes) {
        for (j in 1:n_classes) {
            mus[i,j] ~ uniform(0,1);
            logetas[i,j] ~ logistic(log(100), 1);
        }
    }

    for (c in 1:n_children) {
        for (i in 1:n_classes) {
            for (j in 1:n_classes) {
                child_confusion[c,i,j] ~ beta(alphas[i,j], betas[i,j]);
            }
        }
    }
}
"""

num_chains = 4

posterior = stan.build(stan_code, data = data)
fit = posterior.sample(num_chains = num_chains, num_samples = 4000)
df = fit.to_frame()
df.to_csv('fit.csv')
