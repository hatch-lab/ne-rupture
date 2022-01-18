# coding=utf-8

"""
Classifies particle events by using fixed thresholds

Usage:
  classify.py fixed-cutoff [options] [--] INPUT

Arguments:
  INPUT Path to the directory containing particle data

Options:
  -h, --help
  -v, --version

Output:
  Generates graphs of each nucleus's predicted and actual events.
  Generates annotated videos of each nucleus with either a predicted or a true event.
"""

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

import math
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from lib.summarize import get_cell_stats

NEED_TIFFS = False
SAVES_INTERMEDIATES = False
NAME = "fixed-cutoff"
CONF_PATH = (ROOT_PATH / ("classifiers/fixed-cutoff/conf.json")).resolve()

def get_schema():
  return {
    'fixed-cutoff': lambda x: True
  }

def process_event_seeds(p_data, conf):
  """
  Classify the events in a particle

  - Seeds mitosis/rupture events
  - Find the end of the event (when intensity returns to baseline)
  - Find when recovery/repair begins

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    Panda DataFrame
  """
  idx = (p_data['event'] != 'N')
  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()

  # Find event ends
  p_data = extend_events(p_data, conf)

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, conf)

  return p_data

def seed_events(data, conf):
  """
  Seeds rupture events for full classification

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    Panda DataFrame The modified particle data
  """

  # Filter out particles that are too near each other
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(
    lambda x: np.min(x['nearest_neighbor_distance']) >= 50*x['x_conversion'].iloc[0]
  )

  # Filter out particles that are too jumpy
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: np.all(x['speed'] <= 0.05)) # Faster than 0.05 µm/3 min

  data.sort_values([ 'data_set', 'particle_id', 'frame' ], inplace=True)
  data.reset_index(inplace=True, drop=True)

  r_idx = ((data['median_derivative'] <= conf['median_derivative_cutoff']) & (data['stationary_median'] <= conf['median_cutoff']))
  data.loc[r_idx, 'event'] = 'R'

  return data

def extend_events(p_data, conf):
  """
  Extend seeded events

  For each event, will extend the event out till 
  baseline intensity is reached. Since this will
  in turn affect the baseline intensity, this is
  iterated for each event till convergence is reached.

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Configuration options 

  Returns
    Panda DataFrame The modified data frame
  """
  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()
  convergence_limit = conf['convergence_limit']

  for event_id in event_ids:
    no_event_idx = (p_data['event'] == 'N')
    event_type = p_data[(p_data['event_id'] == event_id)]['event'].values[0]
    baseline_median = np.mean(p_data.loc[no_event_idx, 'stationary_median']) if no_event_idx.any() else 0
    old_baseline_median = 1E6

    start = None

    while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
      # Look forward
      end = np.max(p_data.loc[(p_data['event_id'] == event_id), 'frame'])

      stop_idx = (
        (p_data['frame'] > end) &
        (p_data['stationary_median'] > baseline_median*conf['baseline_median_scale_factor'])
      )

      if not stop_idx.any():
        # There are no frames beyond end that reach baseline
        stop = np.max(p_data['frame'])+1
      else:
        # Get the first frame that satisfies the above conditions
        stop = np.min(p_data.loc[stop_idx, 'frame'])

      # We don't want to run over any other events that have already been seeded
      next_event_idx = (
        (p_data['frame'] < stop) & 
        (p_data['event_id'] != -1) & 
        (p_data['event_id'] != event_id)
      )
      if next_event_idx.any():
        stop = np.min(p_data.loc[next_event_idx, 'frame'])

      # Look backward
      begin = np.min(p_data.loc[(p_data['event_id'] == event_id), 'frame'])
      start_idx = (
        (p_data['frame'] < begin) &
        (p_data['stationary_median'] > baseline_median*conf['baseline_median_scale_factor'])
      )

      if not start_idx.any():
        # There are no frames before begin that reach baseline
        start = np.min(p_data['frame'])
      else:
        # Get the first frame that satisfies the above conditions
        start = np.max(p_data.loc[start_idx, 'frame'])

      # We don't want to run over any other events that have already been seeded
      prev_event_idx = (
        (p_data['frame'] > start) &
        (p_data['event_id'] != -1) &
        (p_data['event_id'] != event_id)
      )
      if prev_event_idx.any():
        start = np.max(p_data.loc[prev_event_idx, 'frame'])

      idx = (
        (p_data['frame'] >= start) &
        (p_data['frame'] < stop)
      )
      p_data.loc[idx, 'event'] = event_type
      p_data.loc[idx, 'event_id'] = event_id

      old_baseline_median = baseline_median
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])
  
  return p_data

def find_event_recoveries(p_data, conf):
  """
  Find recoveries

  We will attempt to use a smoothed median to see
  when the median begins to increase. If none is found,
  we'll use the actual median.

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Configuration options 

  Returns
    Panda DataFrame The modified data frame
  """
  col_names = list(p_data.columns.values)

  # Get a smoothed median intensity
  frame_rate = p_data['frame_rate'].unique()[0]
  p_data.loc[:, 'smoothed'] = sliding_average(p_data['stationary_median'], conf['sliding_window_width'], conf['sliding_window_step'], frame_rate)

  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    # Find where we start returning to baseline, preferentially using the
    # smoothed value, but the normalized value if we have to

    event = p_data.loc[(p_data['event_id'] == event_id),:]
    event_type = event['event'].iloc[0]
    recovery_label = conf['recoveries'][event_type] if event_type in conf['recoveries'] else event_type
    
    idx = (event['smoothed'].diff() > 0)

    if not idx.any():
      idx = (event.loc[:, 'stationary_median'].diff()) > 0

    if idx.any():
      recovery_start = np.min(event.loc[idx, 'time'])
      event_start = np.min(event.loc[:,'time'])
      p_data.loc[((p_data['time'] >= recovery_start) & (p_data['time'] != event_start) & (p_data['event_id'] == event_id)), 'event'] = recovery_label

  return p_data.loc[:, col_names]

def sliding_average(data, window, step, frame_rate):
  """
  Generate a sliding window average

  Arguments:
    data Panda Series The data
    window int The size of the window in seconds
    step int The length the window should slide in seconds
    frame_rate int The number of seconds/frame

  Returns
    Panda Series The modified series
  """

  # window and step are given in seconds
  # We need them in frames
  window = int(window/frame_rate)
  step = int(step/frame_rate)
  total = data.size
  spots = np.arange(1, (total-window)+step, step)
  result = [ 0.0 ]*total

  for i in range(0,len(spots)-1):
    result[spots[i]:(spots[i]+window+1)] = [ np.mean(data.iloc[spots[i]:(spots[i]+window+1)]) ] * len(result[spots[i]:(spots[i]+window+1)])

  return result

def print_progress_bar(message, total, num_left):
  """
  Prints a nice progress bar onto the terminal

  Arguments:
    message string The message to print alongside the bar
    total int The number of child processes to run
    num_left int The number of child processes left to run
  """
  progress = int(30*(total-num_left)//total)
  bar = "#" * progress + ' ' * (30 - progress)
  print("\r{} |{}|".format(message, bar), end="\r")

  if num_left == 0:
    print()

# def find_neighbor_births(n_data, data):
#   data_set = n_data['data_set'].values[0]
#   nearest_neighbor = n_data['nearest_neighbor'].values[0]
#   neighbor_idx = ( (data['data_set'] == data_set) & (data['particle_id'] == nearest_neighbor) )
#   nearest_neighbor_birth = np.min(data.loc[neighbor_idx, 'time'])
#   n_data.loc[:,'nearest_neighbor_birth'] = nearest_neighbor_birth
#   n_data.loc[:,'nearest_neighbor_delta'] = n_data['time']-nearest_neighbor_birth

#   return n_data

def run(data, tiff_path, conf=False, fast=False):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  orig_data = data.copy()

  # Find nearest neighbor distances
  # if not fast:
  #   data = apply_parallel(data.groupby([ 'data_set', 'frame' ]), "Finding nearest neighbors", find_nearest_neighbor_distances)
  #   data = apply_parallel(data.groupby([ 'data_set', 'nearest_neighbor']), "Finding nearest neighbor births", find_neighbor_births, orig_data)

  # Classify
  data.loc[:,'event'] = 'N'
  data.loc[:,'event_id'] = -1
  data = seed_events(data, conf)
  # p_data = seed_mitoses(p_data, conf) # Just skip for now

  if not fast:
    tqdm.pandas(desc="Classifying events")
    data = data.groupby([ 'data_set', 'particle_id' ]).progress_apply(process_event_seeds, conf=conf)

  return ( True, data )

def get_event_summary(data, conf=False):
  """
  Produce an event summary

  Arguments:
    data pd.DataFrame The classified data
    conf bool|dict The conf to be used; False if the default conf file

  Returns:
    pd.DataFrame The event data.
  """
  return False # Not implemented yet

def get_cell_summary(data, conf=False):
  """
  Produce a cell summary

  Arguments:
    data pd.DataFrame The classified data
    conf bool|dict The conf to be used; False if the default conf file

  Returns:
    pd.DataFrame The cell data.
  """
  return data.groupby([ 'data_set', 'particle_id' ]).apply(get_cell_stats)
