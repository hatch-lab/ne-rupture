# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

import math
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
from time import sleep
from PIL import Image, ImageDraw, ImageFont
import cv2

import common.video as hatchvid

NEED_TIFFS = True
NAME = "manual"
CONF_PATH = (ROOT_PATH / ("classifiers/manual/conf.json")).resolve()
FONT_PATH = (ROOT_PATH / ("common/font.ttf")).resolve()

def process_event_seeds(p_data, conf):
  """
  Classify the events in a particle

  - Seeds mitosis/rupture events
  - Find the end of the event (when intensity returns to baseline)
  - Find when recovery/repair begins

  Arguments:
    p_data pd.DataFrame The particle data
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    pd.DataFrame
  """
  idx = (p_data['event'] != 'N')
  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()

  # Find event ends
  p_data = extend_events(p_data, conf)

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, conf)

  return p_data


def seed_events(data, tiff_path, conf):
  """
  Seeds rupture events for full classification

  Arguments:
    p_data pd.DataFrame The particle data
    tiff_path string|Path Path to the TIFFs
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    pd.DataFrame The modified particle data
  """

  title_font = ImageFont.truetype(str(FONT_PATH), size=15)
  small_font = ImageFont.truetype(str(FONT_PATH), size=10)
  p_event_labels = {
    'N': '',
    'R': 'Rupture',
    'M': 'Mitosis',
    'X': 'Apoptosis'
  }

  data.sort_values([ 'data_set', 'particle_id', 'frame' ], inplace=True)
  data.reset_index(inplace=True, drop=True)

  idx = 0
  while(idx < len(data.index)):
    if idx < 0:
      idx = 0

    data_set = data['data_set'].iloc[idx]
    current_event = data['event'].iloc[idx]
    frame = data['frame'].iloc[idx]

    frame_file_name = str(frame).zfill(4) + '.tif'
    frame_path = (tiff_path / (data_set + "/" + frame_file_name)).resolve()

    raw_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

    x = int(round(data['x'].iloc[idx]/data['x_conversion'].iloc[idx]))
    y = int(round(data['y'].iloc[idx]/data['y_conversion'].iloc[idx]))

    # Crop down to just this particle
    crop_frame = hatchvid.crop_frame(raw_frame, x, y, conf['movie_width'], conf['movie_height'])
    crop_frame = cv2.resize(crop_frame, None, fx=conf['movie_scale_factor'], fy=conf['movie_scale_factor'], interpolation=cv2.INTER_CUBIC)

    # Add a space for text
    crop_frame = np.concatenate((np.zeros(( 40, crop_frame.shape[1]), dtype=crop_frame.dtype), crop_frame), axis=0)

    # Make frame text
    title = data['particle_id'].iloc[idx]

    # Describe the current event state
    title += " " + p_event_labels[current_event]

    time = data['time'].iloc[idx]

    hours = math.floor(time / 3600)
    minutes = math.floor((time - (hours*3600)) / 60)
    seconds = math.floor((time - (hours*3600)) % 60)

    label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, frame)

    # Now add text
    img = Image.fromarray(crop_frame)
    draw = ImageDraw.Draw(img)

    controls = "[<-] Prev [->] Next [S] Skip \n[R] Rupture [M] Mitosis [X] Apoptosis [N] None"

    draw.text((10, 10), title, fill='rgb(255,255,255)', font=title_font)
    draw.text((10, 30), label, fill='rgb(255,255,255)', font=small_font)
    draw.text((10, crop_frame.shape[0]-40), controls, fill='rgb(255,255,255)', font=small_font)

    # Get it back into OpenCV format
    crop_frame = np.array(img)

    cv2.imshow('Cell', crop_frame)

    c = cv2.waitKey(0)
    if c == 2:
      idx -= 1

    elif c == 3:
      idx += 1

    elif c == 0:
      idx -= 3

    elif c == 1:
      idx += 3

    elif chr(c & 255) == 'p':
      # Just skip to the prev cell
      particle_ids = data.loc[(data['data_set'] == data_set), 'particle_id'].unique().tolist()
      particle_id = data['particle_id'].iloc[idx]
      prev_particle_id_i = particle_ids.index(particle_id)-1

      if prev_particle_id_i < 0:
        data_sets = data['data_set'].unique().tolist()
        prev_data_set_i = data_sets.index(data_set)-1

        if prev_data_set_i < 0:
          # We're at the beginning
          idx = 0
          continue

        prev_data_set = data_sets[prev_data_set_i]
        idx = np.max(data.index[(data['data_set'] == prev_data_set)].tolist())

      else:
        prev_particle_id = particle_ids[prev_particle_id_i]
        idx = np.max(data.index[( (data['data_set'] == data_set) & (data['particle_id'] == prev_particle_id) )].tolist())

    elif chr(c & 255) == 's':
      # Just skip the rest of this cell
      particle_ids = data.loc[(data['data_set'] == data_set), 'particle_id'].unique().tolist()
      particle_id = data['particle_id'].iloc[idx]
      next_particle_id_i = particle_ids.index(particle_id)+1

      if next_particle_id_i >= len(particle_ids):
        data_sets = data['data_set'].unique().tolist()
        next_data_set_i = data_sets.index(data_set)+1

        if next_data_set_i >= len(data_sets):
          # We're done
          break

        next_data_set = data_sets[next_data_set_i]
        idx = np.min(data.index[(data['data_set'] == next_data_set)].tolist())

      else:
        next_particle_id = particle_ids[next_particle_id_i]
        idx = np.min(data.index[( (data['data_set'] == data_set) & (data['particle_id'] == next_particle_id) )].tolist())

    elif chr(c & 255) in [ 'r', 'm', 'x', 'n' ]:
      data.at[idx, 'event'] = chr(c & 255).upper()

    if 'q' == chr(c & 255):
      exit()

  cv2.destroyAllWindows()

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
        (p_data['event_id'] == (event_id+1))
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
        (p_data['frame'] >= start) &
        (p_data['event_id'] != -1) &
        (p_data['event_id'] == (event_id-1))
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

    if recovery_label == event_type:
      # We don't need to do anything
      continue
    
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

def apply_parallel(grouped, message,  fn, *args):
  """
  Function for parallelizing particle classification

  Will take each DataFrame produced by grouping by particle_id
  and pass that data to the provided function, along with the 
  supplied arguments.

  Arguments:
    grouped list List of grouped particle data
    message string The message to print alongside the loading bars
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
      print_progress_bar(message, total, rs._number_left)
      sleep(2)

  print_progress_bar(message, total, 0)
  return pd.concat(rs.get(), sort=False)

def run(data, tiff_path, conf=False, fast=False):
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  orig_data = data.copy()

  # Classify
  data.loc[:,'event'] = 'N'
  data.loc[:,'event_id'] = -1
  data = seed_events(data, tiff_path, conf=conf)

  if not fast:
    data = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), "Classifying particles", process_event_seeds, conf)

  return data