#!/usr/bin/env python3
# find ../../data/solomon-ana/output/clips -maxdepth 1 -type f -exec sbatch -c 2 --mem 1G -o vtc ./apply.sh {} --device=cpu \;

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
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
import soundfile

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

def process_clip(am, annotator, clip):
    index, ann = clip
    recording_filename, range_onset, range_offset, clip_path = index

    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    vtc_speakers = ['KCHI', 'CHI', 'FEM', 'MAL']

    data = None
    for i, speaker in enumerate(vtc_speakers):
        signal, sr = librosa.load(
            clip_path + f'/{speaker}.wav',
            sr = None
        )
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

        s = []
        for explained in vtc[f'{speaker_A}_vocs_explained']:
            onset, offset = explained
            onset -= range_onset
            offset -= range_onset
            
            s.append(np.mean(data[i, int(onset*time_scale):int(offset*time_scale)]))
        stats[f'{speaker_A}_vocs_explained'] = s

        for j, speaker_B in enumerate(speakers):
            sA = []
            sB = []

            for fp in vtc[f'{speaker_A}_vocs_fp_{speaker_B}']:  
                onset, offset = fp
                onset -= range_onset
                offset -= range_onset

                sA.append(np.mean(data[i, int(onset*time_scale):int(offset*time_scale)]))
                sB.append(np.mean(data[j, int(onset*time_scale):int(offset*time_scale)]))

            stats[f'{speaker_A}_vocs_fp_{speaker_B}_false'] = sA
            stats[f'{speaker_A}_vocs_fp_{speaker_B}_true'] = sB

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

    with mp.Pool(processes = 4) as pool:
        stats = [
            process_clip(am, annotator, clip)
            for clip in intersection.groupby([
                'recording_filename', 'range_onset', 'range_offset', 'clip_path'
            ])
        ]

    return stats
    
if __name__ == '__main__':
    annotators = pd.read_csv('input/annotators.csv')[0:1]
    annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
    stats = [generate_stats(annotator) for annotator in annotators.to_dict(orient = 'records')]
    stats = sum(stats, [])

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
