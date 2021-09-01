#!/usr/bin/env python3

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.metrics import AclewMetrics

import argparse

import datalad.api
from os.path import join as opj
from os.path import basename, exists

import multiprocessing as mp

import numpy as np
import pandas as pd
import pickle

from scipy.stats import gamma
from matplotlib import pyplot as plt

def rates(parameters):
    corpus = parameters["corpus"]
    annotator = parameters["annotator"]
    speakers = ["CHI", "OCH", "FEM", "MAL"]

    project = ChildProject(parameters["path"])
    am = AnnotationManager(project)
    am.read()

    pipeline = AclewMetrics(
        project,
        vtc=annotator,
        alice=None,
        vcm=None,
        from_time="09:00",
        to_time="18:00",
        by="child_id",
    )
    metrics = pipeline.extract()
    print(metrics)
    return pd.DataFrame(metrics).assign(corpus=corpus)

if __name__ == "__main__":
    annotators = pd.read_csv("input/annotators.csv")
    annotators = annotators[~annotators['annotator'].str.startswith('eaf_2021')]
    annotators["path"] = annotators["corpus"].apply(lambda c: opj("input", c))

    with mp.Pool(processes=8) as pool:
        rates = pd.concat(pool.map(rates, annotators.to_dict(orient="records")))

    rates.dropna(inplace = True)
    
    params = []
    speakers = ['CHI', 'OCH', 'FEM', 'MAL']
    for speaker in speakers:
        data = rates[f"voc_{speaker.lower()}_ph"]*9
        a, loc, scale = gamma.fit(data, floc=0)

        x = np.linspace(0, data.max(), 100)
        y = gamma.pdf(x, a, loc, scale)

        plt.cla()
        plt.clf()

        plt.hist(data, bins = 10, density = True)
        plt.plot(x, y)
        plt.savefig(f'output/dist_{speaker}.png')

        params.append({
            'alpha': a,
            'beta': 1/scale,
            'speaker': speaker
        })

    pd.DataFrame(params).set_index('speaker').to_csv('output/speech_dist.csv')
