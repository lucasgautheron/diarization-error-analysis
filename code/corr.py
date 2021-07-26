from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import datalad.api
from os.path import join as opj
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline

import stan
import numpy as np
from scipy.stats import beta

# def extrude(self, removed, mode: str = 'intersection'):
#     if isinstance(removed, Segment):
#         removed = Timeline([removed])

#     truncating_support = removed.gaps(support=self.extent())
#     # loose for truncate means strict for crop and vice-versa
#     if mode == "loose":
#         mode = "strict"
#     elif mode == "strict":
#         mode = "loose"
    
#     return self.crop(truncating_support, mode=mode)

def compute_counts(parameters):
    annotator = parameters['annotator']
    speakers = ['CHI', 'FEM', 'MAL', 'OCH']

    project = ChildProject(parameters['path'])
    am = AnnotationManager(project)
    am.read()

    intersection = AnnotationManager.intersection(
        am.annotations, ['vtc', annotator]
    )

    intersection['path'] = intersection.apply(
        lambda r: opj(project.path, 'annotations', r['set'], 'converted', r['annotation_filename']),
        axis = 1
    )
    datalad.api.get(list(intersection['path'].unique()))

    segments = am.get_collapsed_segments(intersection)
    segments['set'] = segments['set'].replace({annotator: 'truth'})
    segments = segments[segments['segment_offset'] > segments['segment_onset'] + 100]
    segments = segments[segments['speaker_type'].isin(speakers)]

    return (
        segments
        .groupby(['set', 'recording_filename', 'range_onset', 'speaker_type'])
        .agg(
            count = ('segment_onset', 'count')
        )
        .reset_index()
        .pivot(index = ['recording_filename', 'range_onset'], columns = ['set', 'speaker_type'], values = ['count'])
        .assign(
            corpus = parameters['corpus']
        )
    )

    # vtc = {
    #     speaker: segments_to_annotation(segments[(segments['set'] == 'vtc') & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
    #     for speaker in speakers
    # }

    # truth = {
    #     speaker: segments_to_annotation(segments[(segments['set'] == annotator) & (segments['speaker_type'] == speaker)], 'speaker_type').get_timeline()
    #     for speaker in speakers
    # }

    # for speaker_A in speakers:
    #     vtc[f'{speaker_A}_vocs_explained'] = vtc[speaker_A].crop(truth[speaker_A], mode = 'loose')
    #     vtc[f'{speaker_A}_vocs_fp'] = extrude(vtc[speaker_A], vtc[f'{speaker_A}_vocs_explained'])
    #     vtc[f'{speaker_A}_vocs_fn'] = extrude(truth[speaker_A], truth[speaker_A].crop(vtc[speaker_A], mode = 'loose'))

    #     for speaker_B in speakers:
    #         vtc[f'{speaker_A}_vocs_fp_{speaker_B}'] = vtc[f'{speaker_A}_vocs_fp'].crop(truth[speaker_B], mode = 'loose')

    # confusion_matrix = np.zeros((len(speakers), len(speakers)))
    # uncertainty_matrix = np.zeros((len(speakers), len(speakers), 2))

    # for i, speaker_A in enumerate(speakers):
    #     for j, speaker_B in enumerate(speakers):
    #         if i != j:
    #             z = len(vtc[f'{speaker_A}_vocs_fp_{speaker_B}'])
    #             N = len(truth[speaker_B])
    #             confusion_matrix[i,j] = z/N

    #             a = z
    #             b = N-z
    #             uncertainty_matrix[i,j,0] = beta.ppf(0.025, a, b)
    #             uncertainty_matrix[i,j,1] = beta.ppf(0.975, a, b)

    #         else:
    #             z = len(truth[speaker_A]) - len(vtc[f'{speaker_A}_vocs_fn'])
    #             N = len(truth[speaker_A])
    #             confusion_matrix[i,i] = z/N

    #             a = z
    #             b = N-z
    #             uncertainty_matrix[i,j,0] = beta.ppf(0.025, a, b)
    #             uncertainty_matrix[i,j,1] = beta.ppf(0.975, a, b)
    
    # print(parameters['corpus'], annotator)
    # print(confusion_matrix)
    # print(uncertainty_matrix)

    # vtc = {key: len(vtc[key]) for key in vtc}
    # return vtc


annotators = pd.read_csv('input/annotators.csv')
annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
counts = pd.concat([compute_counts(annotator) for annotator in annotators.to_dict(orient = 'records')])
counts = counts.fillna(0)

truth = np.transpose([counts['count']['truth'][speaker].values for speaker in ['CHI', 'OCH', 'FEM', 'MAL']]).astype(int)
vtc = np.transpose([counts['count']['vtc'][speaker].values for speaker in ['CHI', 'OCH', 'FEM', 'MAL']]).astype(int)

data = {
    'n_clips': truth.shape[0],
    'n_classes': truth.shape[1],
    'truth': truth,
    'vtc': vtc
}

stan_code = """
data {
  int<lower=1> n_clips;   // number of clips
  int<lower=1> n_classes; // number of classes
  int truth[n_clips,n_classes];
  int vtc[n_clips,n_classes];
}
parameters {
  //matrix<lower=1>[n_classes,n_classes] alphas;
  //matrix<lower=1>[n_classes,n_classes] betas;
  matrix<lower=0,upper=1>[n_classes,n_classes] confusion;
}
transformed parameters {
  matrix[n_clips, n_classes] log_lik;
  log_lik = rep_matrix(0, n_clips, n_classes);

  for (k in 1:n_clips) {
    for (i in 1:n_classes) {
        for (chi in 0:truth[k,1]) {
            for (och in 0:truth[k,2]) {
                for (fem in 0:truth[k,3]) {
                    for (mal in 0:truth[k, 4]) {
                        if (mal+fem+och+chi == vtc[k,i]) {
                            log_lik[k,i] += binomial_lpmf(mal | truth[k,4], confusion[4,i]);
                            log_lik[k,i] += binomial_lpmf(fem | truth[k,3], confusion[3,i]);
                            log_lik[k,i] += binomial_lpmf(och | truth[k,2], confusion[2,i]);
                            log_lik[k,i] += binomial_lpmf(chi | truth[k,1], confusion[1,i]);
                        }
                    }
                }
            }   
        }
    }
  }
}
model {
   target += sum(log_lik);

    for (i in 1:n_classes) {
        for (j in 1:n_classes) {
            //alphas[i,j] ~ uniform(1, 100);
            //betas[i,j] ~ uniform(1, 100);
            confusion[i,j] ~ uniform(0, 1);
        }
    }
}
"""

init = {
    'alphas': np.full((truth.shape[1], truth.shape[1]), 1.01),
    'betas': np.full((truth.shape[1], truth.shape[1]), 1.01)
}

num_chains = 4

posterior = stan.build(stan_code, data = data)
fit = posterior.sample(num_chains = num_chains, num_samples = 4000)
df = fit.to_frame()
df.to_csv('fit.csv')