# coding=utf-8

import sys
import os
from pathlib import Path
from importlib import import_module

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count

NAME = "weighted-metaclassifier"
CONF_PATH = (ROOT_PATH / ("classifiers/weighted-metaclassifier/conf.json")).resolve()

STATS_EVENT_MAP = {
  "R": "Rupture",
  "M": "Mitosis",
  "X": "Apoptosis"
}

def apply_parallel(grouped, fn, *args):
  """
  Function for parallelizing particle classification

  Will take each DataFrame produced by grouping by particle_id
  and pass that data to the provided function, along with the 
  supplied arguments.

  Arguments:
    grouped List of grouped particle data
    fn function The function called with a group as a parameter
    args Arguments to pass through to fn

  Returns:
    Pandas DataFrame The re-assembled data.
  """
  with Pool(cpu_count()) as p:
    groups = []
    for name, group in grouped:
      t = tuple([ group ]) + tuple(args)
      groups.append(t)
    chunk = p.starmap(fn, groups)

  return chunk

def get_cell_event(group, classifier_name):
  """
  
  """
  for event,name in STATS_EVENT_MAP.items():
    group.loc[:, classifier_name + '_cell_' + event + '_score'] = group['cell_' + event + '_score'].unique()[0]

  return group

def run(data, conf=False, fast=True):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  validation_data_path = Path(ROOT_PATH / conf['validation_data_path'])

  classified_data = data.copy()

  classified_data.loc[:,'event'] = 'N'
  classified_data.loc[:,'event_id'] = -1
  classified_data.loc[:,'cell_event'] = 'N'

  classifiers = get_classifiers()

  score_columns = {}
  score_weights = {}
  for classifier in classifiers:

    weight = conf[classifier.NAME] if classifier.NAME in conf else 1

    # Get new columns
    for event,name in STATS_EVENT_MAP.items():
      if event not in score_columns:
        score_columns[event] = []
        score_weights[event] = []
      score_columns[event].append(classifier.NAME + '_cell_' + event + '_score')
      score_weights[event].append(weight)

    classified_data = classifier.get_weights(classified_data, validation_data_path)
    classified_data = classified_data.groupby([ 'data_set', 'particle_id' ]).apply(get_cell_event, classifier.NAME)

  for event,name in STATS_EVENT_MAP.items():
    classified_data.loc[:, 'avg_cell_' + event + '_score'] = np.average(classified_data[score_columns[event]], axis=1, weights=score_weights[event])
    classified_data.loc[(classified_data['avg_cell_' + event + '_score'] >= conf['score_cutoff']), 'cell_event'] = event

  return classified_data


def get_classifiers():
  classifier_names = ["fixed-cutoff", "masks", "time-smash"]
  classifiers = []

  for classifier_name in classifier_names:
    classifiers.append(import_module("classifiers." + classifier_name))

  return classifiers