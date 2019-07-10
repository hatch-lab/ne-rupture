# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt

import math
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
from time import sleep

NAME = "time-smash"
CONF_PATH = (ROOT_PATH / ("classifiers/time-smash/conf.json")).resolve()

def classify_cells(data, conf):
  """
  Classify cells based on their MIP over time

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding ruptured cells

  Returns:
    Panda DataFrame The modified particle data
  """

  r_idx = ((data['mip_normalized_sum'] >= conf['mip_cutoff']))
  data.loc[r_idx, 'cell_event'] = 'R'

  return data

def run(data, conf=False, fast=False):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  orig_data = data.copy()

  # Classify
  data.loc[:,'event'] = 'N'
  data.loc[:,'event_id'] = -1
  data.loc[:,'cell_event'] = 'N'
  data = classify_cells(data, conf)

  return data

def get_weights(data, validation_data_path):
  validation_data = pd.read_csv(str(validation_data_path), header=0, dtype={ 'particle_id': str })

  denominators = {}
  for event in validation_data['true_event'].unique():
    num_all_cells = validation_data.loc[(validation_data['true_event'] == event), : ].groupby([ 'data_set', 'particle_id' ]).size().shape[0]
    denominators[event] = num_all_cells
    data.loc[:, 'cell_' + event + '_score'] = 0.0

  data = data.groupby([ 'data_set', 'particle_id' ]).apply(get_cell_weights, validation_data, denominators)

  return data

def get_cell_weights(p_data, validation_data, denominators):
  threshold = np.max(p_data['mip_normalized_sum'])

  for event in validation_data['true_event'].unique():
    num_cells = validation_data.loc[((validation_data['true_event'] == event) & ( validation_data['mip_normalized_sum'] >= threshold)), :].groupby([ 'data_set', 'particle_id' ]).size().shape[0]
    p_data.loc[:, 'cell_' + event + '_score'] = num_cells/denominators[event]

  return p_data

