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
from scipy.stats import iqr
from scipy import spatial
from multiprocessing import Pool, cpu_count
from time import sleep

NAME = "outliers"
CONF_PATH = (ROOT_PATH / ("classifiers/" + NAME + "/conf.json")).resolve()

def classify_particle_events(p_data, conf, convergence_limit = 3E-6):
  """
  Classify the events in a particle

  - Seeds events
  - Distinguish rupture events from other oddities
  - Find the end of the event (when intensity returns to baseline)
  - Find when recovery/repair begins

  Arguments:
    p_data Panda DataFrame The particle data
    convergence_limit float Convergence limit; see description 

  Returns:
    Panda DataFrame
  """

  p_data['filtered'] = 0

  p_data.loc[:,'event'] = 'N'
  p_data.loc[:,'event_id'] = -1

  # Seed event
  p_data = seed_events(p_data, conf)

  # Give an ID to each event
  idx = (p_data['event'] != 'N')
  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()

  # Find event ends
  p_data = extend_events(p_data, conf)

  # Figure out what the events are
  p_data = classify_events(p_data, conf)

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, conf)

  # Filter out events that were captured incorrectly
  p_data = filter_events(p_data, conf)

  return p_data

def seed_events(p_data, conf):
  """
  Seeds events for full classification

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    Panda DataFrame The modified particle data
  """

  # percentile doesn't handle NaNs very well
  median_data = p_data['stationary_median'].dropna()
  area_data = p_data['stationary_area'].dropna()

  idx1 = (
    (p_data['stationary_median'] <= conf['median_cutoff']*iqr(median_data)) &
    (p_data['stationary_area'] >= conf['area_cutoff']*iqr(area_data))
  )

  # idx2 = (
  #   (p_data['stationary_median'] < conf['extreme_median_cutoff']*iqr(median_data)) 
  # )

  p_data.loc[( idx1 ), 'event'] = '?'

  return p_data

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
    no_event_idx = (p_data['event'] == 'N')
    baseline_median = np.mean(p_data.loc[no_event_idx, 'stationary_median']) if no_event_idx.any() else 0
    old_baseline_median = 1E6

    start = None

    while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
      # Look forward
      start = np.min(p_data.loc[(p_data['event_id'] == event_id), 'frame'])

      stop_idx = (
        (p_data['frame'] > start) &
        (p_data['stationary_median'] > baseline_median*conf['baseline_median_scale_factor_f'])
      )

      if not stop_idx.any():
        # There are no frames beyond start that reach baseline
        stop = np.max(p_data['frame'])+1
      else:
        # Get the first frame that satisfies the above conditions
        stop = np.min(p_data.loc[stop_idx, 'frame'])

      # Look backward
      start_idx = (
        (p_data['frame'] < start) &
        (p_data['stationary_median'] > baseline_median*conf['baseline_median_scale_factor_r'])
      )

      if start_idx.any():
        # Get the last frame that satisfies the above conditions
        start = np.max(p_data.loc[start_idx, 'frame'])

      idx = (
        (p_data['frame'] > start) &
        (p_data['frame'] < stop)
      )
      p_data.loc[idx, 'event'] = '?'
      p_data.loc[idx, 'event_id'] = event_id

      old_baseline_median = baseline_median
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])

    # Extend backwards 1 frame
    if start is not None:
      idx = (
        (p_data['frame'] == start-1) & 
        (p_data['event_id'] == event_id)
      )
      p_data.loc[idx, 'event'] = '?'
      p_data.loc[idx, 'event_id'] = event_id
  
  return p_data

def classify_events(p_data, conf):
  # Identify rupture events
  p_data = classify_rupture_events(p_data, conf)

  # Identify mitotic events
  p_data = classify_mitotic_events(p_data, conf)

  # Identify cases that are too hard to call
  p_data = classify_unknown_events(p_data, conf)

  return p_data

def classify_mitotic_events(p_data, conf):
  # Find events where the end of the event coincides with the first-time
  # appearance of another cell nearby

  # event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()
  # for event_id in event_ids:
  #   # Get last 3 frames
  #   event_frames = p_data.loc[(p_data['event_id'] == event_id),].sort_values('time')
  #   end_frames = event_frames.tail(3)

  #   data_set = end_frames['data_set'].iloc[0]
  #   pid = end_frames['particle_id'].iloc[0]

  #   # Find any particles nearby; we will use a bounding box instead
  #   # of a circle to make calculations faster/easier
  #   x_conversion = end_frames['x_conversion'].iloc[0]
  #   y_conversion = end_frames['y_conversion'].iloc[0]
  #   min_x = np.min(end_frames['x'])*x_conversion-conf['mitosis']['search_radius']
  #   max_x = np.max(end_frames['x'])*x_conversion+conf['mitosis']['search_radius']
  #   min_y = np.min(end_frames['y'])*y_conversion-conf['mitosis']['search_radius']
  #   max_y = np.max(end_frames['y'])*y_conversion+conf['mitosis']['search_radius']

  #   min_time = np.min(end_frames['time'])
  #   max_time = np.max(end_frames['time'])+conf['mitosis']['time_radius']

  #   neighbors = DATA.loc[( 
  #     (DATA['data_set'] == data_set) & 
  #     (DATA['particle_id'] != pid) & 
  #     (DATA['x'] >= min_x) &
  #     (DATA['x'] <= max_x) &
  #     (DATA['y'] >= min_y) &
  #     (DATA['y'] <= max_y) &
  #     (DATA['time'] >= min_time) &
  #     (DATA['time'] <= max_time)
  #   ),[ 'data_set', 'particle_id' ]]

  #   # The first appearance of a neighbor needs to be within a temporal
  #   # bounding box
  #   for index,neighbor in neighbors.iterrows():
  #     min_appearance_window = max_time - conf['mitosis']['time_radius']
  #     max_appearance_window = max_time

  #     times = DATA.loc[( 
  #       (DATA['data_set'] == neighbor['data_set']) & 
  #       (DATA['particle_id'] == neighbor['particle_id']) 
  #     ), 'time']

  #     first_appearance = np.min(times)

  #     if first_appearance >= min_appearance_window and first_appearance <= max_appearance_window:
  #       # Mitosis event!
  #       p_data.loc[(p_data['event_id'] == event_id), 'event'] = 'M'
  #       # print("M", event_id, neighbor['particle_id'], first_appearance, min_appearance_window)
  #       break

  return p_data

def classify_rupture_events(p_data, conf):
  p_data.loc[(p_data['event'] == '?'), 'event'] = 'R'

  return p_data

def classify_unknown_events(p_data, conf):
  # event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()
  # for event_id in event_ids:
  #   # Get last 3 frames
  #   event_frames = p_data.loc[(p_data['event_id'] == event_id),].sort_values('time')

  #   data_set = event_frames['data_set'].iloc[0]
  #   pid = event_frames['particle_id'].iloc[0]

  #   # Find any particles nearby; we will use a bounding box instead
  #   # of a circle to make calculations faster/easier
  #   x_conversion = event_frames['x_conversion'].iloc[0]
  #   y_conversion = event_frames['y_conversion'].iloc[0]
  #   min_x = np.min(event_frames['x'])*x_conversion-conf['unknown']['search_radius']
  #   max_x = np.max(event_frames['x'])*x_conversion+conf['unknown']['search_radius']
  #   min_y = np.min(event_frames['y'])*y_conversion-conf['unknown']['search_radius']
  #   max_y = np.max(event_frames['y'])*y_conversion+conf['unknown']['search_radius']

  #   min_time = np.min(event_frames['time'])-conf['unknown']['time_radius']
  #   max_time = np.min(event_frames['time'])

  #   neighbors = DATA.loc[( 
  #     (DATA['data_set'] == data_set) & 
  #     (DATA['particle_id'] != pid) & 
  #     (DATA['x'] >= min_x) &
  #     (DATA['x'] <= max_x) &
  #     (DATA['y'] >= min_y) &
  #     (DATA['y'] <= max_y) &
  #     (DATA['time'] >= min_time) &
  #     (DATA['time'] <= max_time)
  #   ),[ 'data_set', 'particle_id' ]]

  #   # If we have neighbors during an event, just ignore it
  #   if len(neighbors) > 0:
  #     p_data.loc[(p_data['event_id'] == event_id), 'event'] = '?'

  return p_data

def filter_events(p_data, conf):
  """
  Filter out events

  Arguments:
    p_data Panda DataFrame The particle data
    conf dict Configuration options 

  Returns
    Panda DataFrame The modified data frame
  """

  # Filter out frames where the median value of all cell events <= 110
  if p_data.loc[((p_data['event_id'] != -1) & (p_data['median'] > 110)),'event_id'].count() <= 0:
    p_data.loc[:,'event'] = 'N'
    p_data.loc[:,'event_id'] = -1
    p_data.loc[:,'filtered'] = 1

  to_remove = np.array([])

  # Filter out frames where the median is 5*iqr above 0
  median_data = p_data['stationary_median'].dropna()
  area_data = p_data['stationary_area'].dropna()

  idx = (
    (p_data['stationary_median'] > 5*iqr(median_data)) |
    (p_data['stationary_area'] < -5*iqr(area_data))
  )
  np.append(to_remove, p_data.loc[idx,'event_id'].unique())

  event_ids = p_data.loc[(p_data['event'] == "R"), 'event_id'].unique()
  for event_id in event_ids:
    events = p_data.loc[(p_data['event_id']) == event_id, 'event'].unique()
    # Remove ruptures for which there are no repair events
    if 'E' not in events:
      np.append(to_remove, [ event_id ])

  to_remove = np.unique(to_remove).tolist()

  p_data.loc[(p_data['event_id'].isin(to_remove)),'event'] = 'N'
  p_data.loc[(p_data['event_id'].isin(to_remove)),'event_id'] = -1

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

def find_nearest_neighbor_distances(f_data):
  """
  Get the nearest neighbors and distances for each particle_id

  Given a dataframe for a given frame/data_set pair, will find the
  nearest neighbor and its distance, using a KDTree

  Arguments:
    f_data Pandas DataFrame The dataframe

  Returns:
    Pands DataFrame The dataframe with the additional columns
  """
  # Build coordinate list
  x = f_data['x'].tolist()
  y = f_data['y'].tolist()
  
  coords = list(zip(x, y))

  # Build KDTree
  tree = spatial.KDTree(coords)

  res = tree.query([ coords ], k=2)
  distances = res[0][...,1][0]
  idxs = res[1][...,1][0]
  neighbor_ids = f_data['particle_id'].iloc[idxs].tolist()
  f_data['nearest_neighbor'] = neighbor_ids
  f_data['nearest_neighbor_distance'] = distances

  return f_data

def run(data, conf=False):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)
      
  # Find the distance to the nearest neighbor
  # groups = data.groupby([ 'data_set', 'frame' ])

  # modified_data = []
  # progress = 0
  # for name,group in groups:
  #   progress = int(30*len(modified_data)/len(groups))
  #   bar = "#" * progress + ' ' * (30 - progress)
  #   print("\rFinding nearest neighbor |%s|" % bar, end="\r")

  #   res = find_nearest_neighbor_distances(group.copy())
  #   modified_data.append(res)

  # modified_data = pd.concat(modified_data)
  # bar = "#" * 30
  # print("\rFinding nearest neighbor |%s|" % bar, end="\r")
  # print()

  # modified_data = DATA.copy()

  # Classify
  data = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), classify_particle_events, conf)

  return data