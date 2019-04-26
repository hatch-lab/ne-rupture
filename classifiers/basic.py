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

NAME = "basic"
CONF_PATH = (ROOT_PATH / ("classifiers/basic/conf.json")).resolve()

def classify_particle_events(p_data, conf, convergence_limit = 3E-6):
  """
  Classify the events in a particle

  - Seeds mitosis/rupture events
  - Find the end of the event (when intensity returns to baseline)
  - Find when recovery/repair begins

  Arguments:
    p_data Panda DataFrame The particle data
    convergence_limit float Convergence limit; see description 

  Returns:
    Panda DataFrame
  """

  idx = (p_data['event'] != 'N')
  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()

  # Find event ends
  p_data = extend_events(p_data, CONF)

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, CONF)

  # Filter out events that were captured incorrectly
  p_data = filter_events(p_data, CONF)

  return p_data

def seed_ruptures(data, conf):
  """
  Seeds rupture events for full classification

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    Panda DataFrame The modified particle data
  """

  idx = ((data['median_derivative'] <= conf['R-median_derivative']) & (data['area_derivative'] >= conf['R-area_derivative']))
  data.loc[idx, 'event'] = 'R'

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
  # print(event_ids)

  for event_id in event_ids:
    # print(p_data['particle_id'].unique()[0], event_id)
    event_type = p_data[(p_data['event_id'] == event_id)]['event'].values[0]
    baseline_scale_factor = event_type + "-baseline_median_scale_factor"
    # print(event_type)

    if baseline_scale_factor not in conf:
      continue

    old_baseline_median = 1E6
    
    no_event_idx = (p_data['event'] == 'N')
    if no_event_idx.any():
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])
    else:
      baseline_median = 0

    while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
      start = np.max(p_data.loc[(p_data['event_id'] == event_id), 'frame'])

      reach_baseline_idx = (
        (p_data['frame'] > start) &
        (p_data['stationary_median'] > baseline_median*conf[baseline_scale_factor])
      )

      if not reach_baseline_idx.any():
        # There are no frames beyond start that reach baseline
        reach_baseline = np.max(p_data['frame'])+1
      else:
        # Get the first frame that satisfies the above conditions
        reach_baseline = np.min(p_data.loc[reach_baseline_idx, 'frame'])

      # print(start, reach_baseline)
      # print(event_id == event_ids[-1])

      # Find the next event
      next_start_idx = (
        (p_data['event_id'] != -1) & 
        (p_data['event_id'] != event_id) & 
        (p_data['frame'] > start)
      )

      if next_start_idx.any():
        # If this isn't the last event in the series, find the start of the
        # next event
        next_start = np.min(p_data.loc[next_start_idx,'frame'])
      else:
        next_start = np.max(p_data['frame'])+1

      # print(next_start)

      stop = np.min([ reach_baseline, next_start ])

      idx = (
        (p_data['frame'] > start) &
        (p_data['frame'] < stop)
      )
      p_data.loc[idx, 'event'] = event_type
      p_data.loc[idx, 'event_id'] = event_id

      old_baseline_median = baseline_median
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])
      # print()

    # Extend backwards 1 frame
    idx = (
      (p_data['frame'] == start-1) & 
      (p_data['event_id'] == event_id)
    )
    p_data.loc[idx, 'event'] = event_type
    p_data.loc[idx, 'event_id'] = event_id
  
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
  p_data.loc[:, 'smoothed'] = sliding_average(p_data['normalized_median'], conf['sliding_window_width'], conf['sliding_window_step'], frame_rate)

  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    # Find where we start returning to baseline, preferentially using the
    # smoothed value, but the normalized value if we have to

    event = p_data.loc[(p_data['event_id'] == event_id),:]
    event_type = event['event'].iloc[0]
    recovery_label = event_type + "-recovery_event_label"
    
    idx = (event['smoothed'].diff() > 0)

    if recovery_label not in conf:
      continue

    if not idx.any():
      idx = (event.loc[:, 'normalized_median'].diff()) > 0

    if idx.any():
      recovery_start = np.min(event.loc[idx, 'time'])
      event_start = np.min(event.loc[:,'time'])
      p_data.loc[((p_data['time'] >= recovery_start) & (p_data['time'] != event_start) & (p_data['event_id'] == event_id)), 'event'] = conf[recovery_label]

  return p_data.loc[:, col_names]

def filter_events(p_data, conf):
  """
  Filter out events

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Configuration options 

  Returns
    Panda DataFrame The modified data frame
  """
  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()
  to_remove = []

  for event_id in event_ids:
    event = p_data.loc[(p_data['event_id'] == event_id),:]

    if 'R' not in event['event'].unique():
      # We're only doing ruptures
      continue

    # Remove ruptures for which there are no repair events
    if 'E' not in event['event'].unique():
      to_remove.append(event_id)

  to_remove = list(set(to_remove))

  p_data.loc[(p_data['event_id'].isin(to_remove)),'event'] = 'N'
  p_data.loc[(p_data['event_id'].isin(to_remove)),'event_id'] = -1

  return p_data

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

def print_progress_bar(total, num_left):
  """
  Prints a nice progress bar onto the terminal

  Arguments:
    total int The number of child processes to run
    num_left int The number of child processes left to run
  """
  progress = int(30*(total-num_left)//total)
  bar = "#" * progress + ' ' * (30 - progress)
  print("\rClassifying particles |%s|" % bar, end="\r")

  if num_left == 0:
    print()

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

  total_groups = len(grouped)

  with Pool(cpu_count()) as p:
    groups = []
    for name, group in grouped:
      t = tuple([ group ]) + tuple(args)
      groups.append(t)
    rs = p.starmap_async(fn, groups)
    total = rs._number_left

    while not rs.ready():
      print_progress_bar(total, rs._number_left)
      sleep(2)

  print_progress_bar(total, 0)
  return pd.concat(rs.get(), sort=False)

def run(data, conf=False, fast=False):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  # Classify
  data.loc[:,'event'] = 'N'
  data.loc[:,'event_id'] = -1
  data = seed_ruptures(data, conf)

  if not fast:
    data = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), classify_particle_events)

  return data

