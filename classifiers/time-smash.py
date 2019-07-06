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