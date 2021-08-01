#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import datalad.api
from os.path import join as opj
from os.path import basename, exists
import pandas as pd

import stan
import numpy as np
from scipy.stats import beta

from matplotlib import pyplot as plt
import seaborn as sns

def compute_counts(parameters):
    annotator = parameters['annotator']
    corpus = parameters['corpus']
    speakers = ['CHI', 'FEM', 'MAL', 'OCH']

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
    segments['child'] = corpus + '_' + segments['child_id']

    segments = segments[segments['speaker_type'].isin(speakers)]

    segments['set'] = segments['set'].replace({annotator: 'truth'})
    segments = segments[segments['segment_offset'] > segments['segment_onset'] + 100]
    
    return (
        segments
        .groupby(['set', 'child', 'recording_filename', 'range_onset', 'speaker_type'])
        .agg(
            count = ('segment_onset', 'count')
        )
        .reset_index()
        .pivot(index = ['child', 'recording_filename', 'range_onset'], columns = ['set', 'speaker_type'], values = ['count'])
        .assign(
            corpus = corpus
        )
    )

annotators = pd.read_csv('input/annotators.csv')
annotators = annotators[0:1]
annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
counts = pd.concat([compute_counts(annotator) for annotator in annotators.to_dict(orient = 'records')])
counts = counts.fillna(0)

truth = np.transpose([counts['count']['truth'][speaker].values for speaker in ['CHI', 'OCH', 'FEM', 'MAL']]).astype(int)
vtc = np.transpose([counts['count']['vtc'][speaker].values for speaker in ['CHI', 'OCH', 'FEM', 'MAL']]).astype(int)

counts.reset_index(inplace = True)
counts['child'] = counts['child'].astype('category').cat.codes

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
    'n_children': counts['child'].nunique(),
    'child': 1+counts['child'].astype('category').cat.codes.values,
    'truth': truth.astype(int),
    'vtc': vtc.astype(int)
}

print(f"clips: {data['n_clips']}")
print(f"children: {data['n_children']}")
print("true vocs: {}".format(np.sum(data['truth'])))
print("vtc vocs: {}".format(np.sum(data['vtc'])))

plt.scatter(data['truth'][:,0]+np.random.normal(0,0.1,truth.shape[0]), data['vtc'][:,0]+np.random.normal(0,0.1,truth.shape[0]))
plt.scatter(data['truth'][:,1]+np.random.normal(0,0.1,truth.shape[0]), data['vtc'][:,1]+np.random.normal(0,0.1,truth.shape[0]))
plt.scatter(data['truth'][:,2]+np.random.normal(0,0.1,truth.shape[0]), data['vtc'][:,2]+np.random.normal(0,0.1,truth.shape[0]))
plt.scatter(data['truth'][:,3]+np.random.normal(0,0.1,truth.shape[0]), data['vtc'][:,3]+np.random.normal(0,0.1,truth.shape[0]))

plt.show()

stan_code = """
data {
  int<lower=1> n_clips;   // number of clips
  int<lower=1> n_children; // number of children
  int<lower=1> n_classes; // number of classes
  int child[n_clips];
  int truth[n_clips,n_classes];
  int vtc[n_clips,n_classes];
}

parameters {
  matrix<lower=0,upper=1>[n_classes,n_classes] mus;
  matrix<lower=0>[n_classes,n_classes] logetas;
  matrix<lower=0,upper=1>[n_classes,n_classes] child_confusion[n_children];
}

transformed parameters {
  matrix<lower=1>[n_classes,n_classes] alphas;
  matrix<lower=1>[n_classes,n_classes] betas;

  matrix[n_clips, n_classes] log_lik;
  log_lik = rep_matrix(0, n_clips, n_classes);

  alphas = mus * exp(logetas);
  betas = (1-mus) * exp(logetas);

    for (k in 1:n_clips) {
        for (i in 1:n_classes) {
            int n = 1;
            vector[200] log_contrib_comb;
            log_contrib_comb = rep_vector(0, 200);
            for (chi in 0:truth[k,1]) {
                for (och in 0:truth[k,2]) {
                    for (fem in 0:truth[k,3]) {
                        for (mal in 0:truth[k, 4]) {
                            if (mal+fem+och+chi == vtc[k,i]) {
                                log_contrib_comb[n] += binomial_lpmf(mal | truth[k,4], child_confusion[child[k],4,i]);
                                log_contrib_comb[n] += binomial_lpmf(fem | truth[k,3], child_confusion[child[k],3,i]);
                                log_contrib_comb[n] += binomial_lpmf(och | truth[k,2], child_confusion[child[k],2,i]);
                                log_contrib_comb[n] += binomial_lpmf(chi | truth[k,1], child_confusion[child[k],1,i]);
                                n = n+1;
                            }
                        }
                    }
                }
            }
            log_lik[k,i] = log_sum_exp(log_contrib_comb[1:n]);
        }
    }
}
model {
    for (k in 1:n_clips) {
        target += log_sum_exp(log_lik[k,:]);
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
