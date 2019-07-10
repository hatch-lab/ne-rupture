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

NAME = "any-metaclassifier"
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

def get_init_ruptures(group):
  """
  
  """
  if 'all_cell_event' in group.columns:
    cell_event = group['all_cell_event'].unique()[0]
  else:
    cell_event = 'N'

  cell_event = 'R' if 'R' in group['event'].unique() else cell_event
  if 'cell_event' in group.columns:
    cell_event = 'R' if 'R' in group['cell_event'].unique() else cell_event

  group.loc[:, 'all_cell_event'] = cell_event

  return group

def run(data, conf=False, fast=True):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  classified_data = data.copy()

  classified_data.loc[:,'event'] = 'N'
  classified_data.loc[:,'event_id'] = -1
  classified_data.loc[:,'cell_event'] = 'N'

  classifiers = get_classifiers()

  # Get initial ruptures by accepting any ruptures that are classified
  for classifier in classifiers:
    classified_data = classifier.run(classified_data, fast=True)
    classified_data = classified_data.groupby([ 'data_set', 'particle_id' ]).apply(get_init_ruptures)

  classified_data = classified_data.loc[(classified_data['all_cell_event'] == 'R'), :]

  classified_data = classifiers[0].run(classified_data, conf=conf, fast=True)

  return classified_data


def get_classifiers():
  classifier_names = ["fixed-cutoff", "masks", "time-smash"]
  classifiers = []

  for classifier_name in classifier_names:
    classifiers.append(import_module("classifiers." + classifier_name))

  return classifiers