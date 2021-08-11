#!/usr/bin/env python3

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

def process_clip(am, clip):
    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    vtc_speakers = ['KCHI', 'CHI', 'FEM', 'MAL']

    onset = clip['range_onset']
    offset = clip['range_offset']

    rows = []
    for i, speaker in enumerate(vtc_speakers):
        signal, sr = librosa.load(
            clip['clip_path'] + f'_{speaker}',
            sr = 16000,
            offset = 2,
            duration = (offset-onset)/1000
        )
        rows.append(signal)

    data = np.array(rows)

    ann = pd.concat([intersection, intersection.assign(set = annotator)])
    segments = am.get_segments(ann)

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
            s.append(np.mean(data[i, 16*onset:16*offset]))

        stats[f'{speaker_A}_vocs_explained'] = s

        for j, speaker_B in enumerate(speakers):
            sA = []
            sB = []

            for fp in vtc[f'{speaker_A}_vocs_fp_{speaker_B}']:  
                onset, offset = fp
                sA.append(np.mean(data[i, 16*onset:16*offset]))
                sB.append(np.mean(data[j, 16*onset:16*offset]))

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
        lambda f: opj('output/scores/', splitext(basename(clip['recording_filename']))[0], str(clip['range_onset']) + '_' + str(clip['range_offset'])),
        axis = 1
    )

    clips = intersection[intersection['set'] == 'vtc']
    with mp.Pool(processes = 32) as mp:
        stats = pool.map(partial(process_clip, am), clips)

    return stats
    
if __name__ == '__main__':
    annotators = pd.read_csv('input/annotators.csv')
    annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
    stats = [generate_stats(annotator) for annotator in annotators.to_dict(orient = 'records')]
