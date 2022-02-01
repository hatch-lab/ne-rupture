# coding=utf-8

"""
Manually classify cells

Usage:
  classify.py manual [options] [--] INPUT

Arguments:
  INPUT Path to the directory containing particle data

Options:
  -h, --help
  -v, --version
  --start-over  Whether to start over
  --distance-filter=<int>  [default: 0] If a given cell is ever closer than this value (in um) to another cell, it is excluded
  --jumpy-filter=<float>  [default: 0.0] If a given cell ever moves faster than this value (in microns/180s), it is excluded

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
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import optimize
import contextlib
from skimage import exposure, util
from tqdm import tqdm

import lib.video as hatchvid
from lib.summarize import get_cell_stats

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

NEED_TIFFS = True
SAVES_INTERMEDIATES = True
NAME = "manual"
CONF_PATH = (ROOT_PATH / ("classifiers/manual/conf.json")).resolve()
FONT_PATH = (ROOT_PATH / ("lib/fonts/font.ttf")).resolve()

def get_schema():
  return {
    'manual': lambda x: True,
    Optional('--start-over'): bool,
    '--distance-filter': And(Use(int), lambda n: n >= 0),
    '--jumpy-filter': And(Use(float), lambda n: n >= 0.0)
  }

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

  # Perform curve-fit for rupture/repair events
  p_data['repair_k'] = np.nan
  p_data = fit_re_curves(p_data)

  # Determine event durations
  p_data = p_data.groupby([ 'event_id', 'event' ]).apply(find_event_durations)

  return p_data

@contextlib.contextmanager
def atomic_write(file_path):
  temp_file_name = str(file_path) + "~"
  with open(temp_file_name, "w") as f:
    yield f
  os.rename(temp_file_name, str(file_path))

def seed_events(data, tiff_path, tmp_csv_path, conf, idx=0):
  """
  Seeds rupture events for full classification

  Arguments:
    p_data pd.DataFrame The particle data
    tiff_path string|Path Path to the TIFFs
    conf dict Cutoffs for various parameters used for finding rupture events

  Returns:
    (bool, pd.DataFrame) Whether seeding completed, The modified particle data
  """
  title_font = ImageFont.truetype(str(FONT_PATH), size=15)
  small_font = ImageFont.truetype(str(FONT_PATH), size=12)
  p_event_labels = {
    'N': '',
    'R': 'Rupture',
    'M': 'Mitosis',
    'X': 'Apoptosis'
  }

  intensity_range = [ 0, 255 ]
  inversion = False
  while(idx < len(data.index)):

    # Bounds checking
    if idx < 0:
      idx = 0

    if intensity_range[0] <= 0:
      intensity_range[0] = 0
    if intensity_range[1] >= 255:
      intensity_range[1] = 255

    data.loc[:, 'current_idx'] = idx

    data_set = data['data_set'].iloc[idx]
    current_event = data['event'].iloc[idx]
    frame = int(data['frame'].iloc[idx])

    frame_file_name = str(frame).zfill(4) + '.tif'
    frame_path = (tiff_path / (data_set + "/" + frame_file_name)).resolve()

    raw_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

    x = data['x_px'].iloc[idx]
    y = data['y_px'].iloc[idx]

    # Crop down to just this particle
    crop_frame = hatchvid.crop_frame(raw_frame, x, y, conf['movie_width'], conf['movie_height'])
    crop_frame = cv2.resize(crop_frame, None, fx=conf['movie_scale_factor'], fy=conf['movie_scale_factor'], interpolation=cv2.INTER_CUBIC)

    # Adjust contrast
    crop_frame = exposure.rescale_intensity(crop_frame, in_range=tuple(intensity_range))

    # Add a space for text
    label_buffer = 45
    crop_frame = np.concatenate((np.zeros(( label_buffer, crop_frame.shape[1]), dtype=crop_frame.dtype), crop_frame), axis=0)

    # Invert
    crop_frame = util.invert(crop_frame) if inversion else crop_frame

    # Make frame text
    title = data['particle_id'].iloc[idx]

    # Describe the current event state
    title += " " + p_event_labels[current_event]

    time = data['time'].iloc[idx]

    hours = math.floor(time / 3600)
    minutes = math.floor((time - (hours*3600)) / 60)
    seconds = math.floor((time - (hours*3600)) % 60)

    label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, frame)

    # Add controls
    img = Image.fromarray(crop_frame)
    draw = ImageDraw.Draw(img)

    controls = "[<] Prev [>] Next [S] Skip [Q] Save & Quit\n[-] Less contrast [+] More contrast [I] Invert\n[R] Rupture [M] Mitosis [X] Apoptosis [N] None"

    text_color = 'rgb(0,0,0)' if inversion else 'rgb(255,255,255)'
    reticle_color = 'rgb(255,255,255)' if inversion else 'rgb(0,0,0)'
    draw.text((10, 10), title, fill=text_color, font=title_font)
    draw.text((10, 30), label, fill=text_color, font=small_font)
    draw.text((10, crop_frame.shape[0]-50), controls, fill=text_color, font=small_font)

    # Add reticles
    center_x = int(crop_frame.shape[1]/2)
    center_y = int((crop_frame.shape[0]+label_buffer)/2)
    draw.line([ (center_x-10, center_y), (center_x+10, center_y) ], fill=reticle_color, width=2)
    draw.line([ (center_x, center_y-10), (center_x, center_y+10) ], fill=reticle_color, width=2)

    # Add progress bar
    width = int(idx/len(data.index)*crop_frame.shape[1])
    draw.rectangle([ (0, 0), (width, 5) ], fill=text_color)

    # Get it back into OpenCV format
    crop_frame = np.array(img)

    cv2.imshow('Cell', crop_frame)

    c = chr(cv2.waitKey(100) & 255)
    if c == '-': # Scale histogram
      intensity_range[1] += 5

    elif c == '=': # Scale histogram
      intensity_range[1] -=5

    elif c == 'i': # Invert
      inversion = ~inversion

    elif c == ',':
      idx -= 1

    elif c == '.':
      idx += 1

    elif c == '<':
      idx -= 3

    elif c == '>':
      idx += 3

    elif c == 'p':
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

    elif c == 's':
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

    elif c in [ 'r', 'm', 'x', 'n' ]:
      data.at[idx, 'event'] = c.upper()

    elif c == 'q' or cv2.getWindowProperty('Cell', cv2.WND_PROP_VISIBLE) < 1:
      cv2.destroyAllWindows()
      return (False, data)

  cv2.destroyAllWindows()
  return (True, data)

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
    event_type = p_data.loc[(p_data['event_id'] == event_id), 'event'].unique()
    if len(event_type) <= 0:
      continue
    else:
      event_type = event_type[0]

    if event_type == 'X':
      frame = np.max(p_data.loc[(p_data['event_id'] == event_id), 'frame'])
      p_data.loc[(p_data['frame'] >= frame), 'event'] = event_type
      p_data.loc[(p_data['frame'] >= frame), 'event_id'] = event_id
      continue

    no_event_idx = (p_data['event'] == 'N')
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

def fit_re_curves(p_data):
  """
  Find import curves

  Attempts to fit a one-phase association 
  curve to every repair event, and find 
  the k parameter.

  Arguments:
    p_data pd.DataFrame The particle data

  Returns
    pd.DataFrame The modified data frame
  """
  baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])

  p_data = p_data.groupby([ 'event_id' ]).apply(fit_re_curve, baseline_median)
  return p_data

def fit_re_curve(pe_data, baseline_median):
  """
  Find import curve for a single repair event

  Attempts to fit a one-phase association curve
  to a single repair event.

  Arguments:
    pe_data pd.DataFrame The particle data
    baseline_media float The baseline median value for non-events

  Returns
    pd.DataFrame The modified data frame
  """
  if pe_data['event'].iloc[0] != 'R' and pe_data['event'].iloc[0] != 'E':
    return pe_data

  if pe_data.loc[( pe_data['event'] == 'E' )].shape[0] <= 1:
    return pe_data

  pe_data.sort_values(by=['time'])

  # Fit curve
  x0 = np.min(pe_data.loc[( pe_data['event'] == 'E' ), 'time'])
  y0 = pe_data.loc[( pe_data['time'] == x0 ), 'stationary_median'].iloc[0]
  top = baseline_median

  x = np.array(pe_data.loc[( pe_data['event'] == 'E' ), 'time'].tolist())
  y = np.array(pe_data.loc[( pe_data['event'] == 'E'), 'stationary_median'].tolist())

  x = x[~np.isnan(y)]
  y = y[~np.isnan(y)]

  try:
    popt, pcov = optimize.curve_fit(lambda time, k: one_phase(time, x0, y0, top, k), x, y, p0=[ 1E-5 ])
  except optimize.OptimizeWarning:
    return pe_data
  except RuntimeError:
    return pe_data

  pe_data.loc[:, 'repair_k'] = popt[0]

  return pe_data

def one_phase(x, x0, y0, top, k):
  """
  A one-phase association curve

  Arguments:
    x np.array The x-values
    x0 float The min x-value
    y0 float The min y-value
    top float The max y-value
    k float The rate constant

  Returns
    np.array The y-values for each x-value
  """
  return y0+(top-y0)*(1-np.exp(-k*(x-x0)))

def find_event_durations(pe_data):
  """
  Find event durations

  Arguments:
    pe_data pd.DataFrame The particle data

  Returns
    pe_data pd.DataFrame The modified data frame
  """
  pe_data.loc[:, 'event_duration'] = np.max(pe_data['time']) - np.min(pe_data['time'])
  return pe_data

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

def run(data, tiff_path, conf=False, fast=False):
  """
  Run the classifier

  Arguments:
    data pd.DataFrame The input data
    tiff_path string The path to TIFF images
    conf bool|dict The conf to be used; False if the default conf file
    fast bool If True, we will not do any processing other than seeding the events

  Returns:
    pd.DataFrame The classified data.
  """
  if not conf:
    with CONF_PATH.open(mode='r') as file:
      conf = json.load(file)

  tmp_path = Path(conf['tmp_path'])
  tmp_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
  
  # Filter out particles that are too near each other
  if conf['--distance-filter'] > 0:
    data = data.groupby([ 'data_set', 'particle_id' ]).filter(
      lambda x: np.min(x['nearest_neighbor_distance']) >= conf['--distance-filter']*x['x_conversion'].iloc[0]
    )

  # Filter out particles that are too jumpy
  if conf['--jumpy-filter'] > 0:
    data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: np.all(x['speed'] <= conf['--jumpy-filter'])) # Faster than 0.05 µm/3 min

  data.sort_values([ 'data_set', 'particle_id', 'frame' ], inplace=True)
  data.reset_index(inplace=True, drop=True)
  idx = data['current_idx'].iloc[0]

  # Classify
  if 'event' not in data.columns:
    data.loc[:,'event'] = 'N'
    data.loc[:,'event_id'] = -1

  done, data = seed_events(data, tiff_path, tmp_path, conf=conf, idx=idx)

  if not fast and done:
    try:
      tqdm.pandas(desc="Processing classified events")
      data = data.groupby([ 'data_set', 'particle_id' ]).progress_apply(process_event_seeds, conf=conf)
    except Exception as e:
      print(e)
      return ( False, data )

  return ( done, data )

def get_event_summary(data, conf=False):
  """
  Produce an event summary

  Arguments:
    data pd.DataFrame The classified data
    conf bool|dict The conf to be used; False if the default conf file

  Returns:
    pd.DataFrame The event data.
  """

  cp = data.copy()

  # Calculate and store baseline median / particle
  cp = cp.groupby([ 'data_set', 'particle_id' ]).apply(get_baseline_median)

  # Filter out non-events
  cp = cp.loc[( cp['event'] != 'N' ), :]

  if cp.shape[0] <= 0:
    return False
  
  tqdm.pandas(desc='Generating event summaries')
  events = cp.groupby([ 'data_set', 'particle_id', 'event_id' ]).progress_apply(get_event_info)
  return events

def get_baseline_median(p_data):
  """
  Get the RAW baseline median intensity for each particle

  Arguments:
    data pd.DataFrame The classified particle data

  Returns:
    pd.DataFrame The data frame with an additional column, raw_baseline_median
  """
  baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'median'])
  p_data.loc[:, 'raw_baseline_median'] = baseline_median

  return p_data

def get_event_info(e_data):
  """
  Get a dataframe summary of each event

  Arguments:
    data pd.DataFrame The classified event data

  Returns:
    pd.DataFrame A summary dataframe for this event
  """
  event_types = e_data['event'].unique().tolist()
  durations = []
  normalized_durations = []
  fractions_fp_lost = []
  rupture_sizes = []

  min_stationary_intensity = np.min(e_data['stationary_median'])
  min_intensity = np.min(e_data['median'])
  baseline_intensity = e_data['raw_baseline_median'].unique()[0]

  for event_type in event_types:
    duration = e_data.loc[( e_data['event'] == event_type ), 'event_duration'].unique()[0]
    durations.append(duration)

    normalized_duration = duration * abs(min_stationary_intensity)
    normalized_durations.append(normalized_duration)

    fractions_fp_lost.append(min_intensity/baseline_intensity)

    first_time = np.min(e_data['time'])
    first_velocity = e_data.loc[( e_data['time'] == first_time ), 'median_derivative'].unique()[0]

    # If this is a rupture, the rupture hole can be estimated by the 
    # following linear relationship:
    # area nm = -0.01249x-0.01079 where x is in  s^-1
    rupture_sizes.append(-0.01249*first_velocity-0.01079)

  if len(durations) > 1:
    duration_sum = np.sum(durations)

    event_types.append("+".join(event_types))
    durations.append(duration_sum)
    normalized_durations.append(duration_sum * abs(min_stationary_intensity))

    fractions_fp_lost.append(min_intensity/baseline_intensity)
    rupture_sizes.append(-0.01249*first_velocity-0.01079)

  data_sets = [ e_data['data_set'].unique()[0] ]*len(event_types)
  particle_ids = [ e_data['particle_id'].unique()[0] ]*len(event_types)
  
  return pd.DataFrame({
    'data_set': data_sets,
    'particle_id': particle_ids,
    'event': event_types,
    'duration': durations,
    'normalized_duration': normalized_durations,
    'fraction_fp_lost': fractions_fp_lost,
    'rupture_size': rupture_sizes
  })

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
