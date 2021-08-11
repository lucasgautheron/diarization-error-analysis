#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.metrics import segments_to_annotation

import datalad.api
from os.path import join as opj
from os.path import basename, exists, splitext
import pandas as pd

import numpy as np

def extract_clips(parameters):
    annotator = parameters['annotator']

    project = ChildProject(parameters['path'])
    am = AnnotationManager(project)
    am.read()

    intersection = AnnotationManager.intersection(
        am.annotations, ['vtc', annotator]
    )

    intersection['recording_path'] = intersection['recording_filename'].apply(
        lambda f: opj(project.path, 'recordings/raw', f)
    )

    datalad.api.get(list(intersection['recording_path'].unique()))

    clips = intersection[intersection['set'] == 'vtc'][['recording_path', 'recording_filename', 'range_onset', 'range_offset']]
    for clip in clips.to_dict(orient = 'records'):
        bn = splitext(basename(clip['recording_filename']))[0]
        onset = clip['range_onset']
        offset = clip['range_offset']

        signal, sr = librosa.load(
            clip['recording_path'],
            sr = 16000,
            offset = (onset-2000)/1000,
            duration = (offset-onset+2000)/1000
        )

        output = f'output/clips/{bn}_{onset}_{offset}.wav'
        soundfile.write(output, signal, 16000)

    return clips
    

annotators = pd.read_csv('input/annotators.csv')
annotators['path'] = annotators['corpus'].apply(lambda c: opj('input', c))
clips = pd.concat([extract_clips(annotator) for annotator in annotators.to_dict(orient = 'records')])
clips.to_csv('output/clips/clips.csv', index = False)