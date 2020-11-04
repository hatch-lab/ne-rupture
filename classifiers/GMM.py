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
from tqdm import tqdm

from scipy import optimize
import common.video as hatchvid
from preprocessors.lib import fit_splines
from validate.lib import get_cell_stats

from sklearn.mixture import GaussianMixture

import joblib

NEED_TIFFS = False
SAVES_INTERMEDIATES = False
NAME = "GMM"
CONF_PATH = (ROOT_PATH / ("classifiers/GMM/conf.json")).resolve()

def get_extremes(event_data, cols):
  """
  Get extreme values (min, max) for the given cols in this event

  Arguments:
    event_data pd.DataFrame The event data to examine
    cols list A list of strings naming the columns for which extreme values should be found

  Returns:
    pd.DataFrame A pandas dataframe in long format
  """
  extreme_data = {
    'data_set': [],
    'particle_id': [],
    'true_event_id': [],
    'variable': [],
    'value': []
  }

  for col in cols:
    extreme_data['variable'].append(col + '_min')
    extreme_data['value'].append(event_data[col].min())
    extreme_data['variable'].append(col + '_max')
    extreme_data['value'].append(event_data[col].max())

  extreme_data['data_set'] = [ event_data['data_set'].iloc[0] ]*len(extreme_data['variable'])
  extreme_data['particle_id'] = [ event_data['particle_id'].iloc[0] ]*len(extreme_data['variable'])
  extreme_data['true_event_id'] = [ event_data['true_event_id'].iloc[0] ]*len(extreme_data['variable'])

  extreme_data = pd.DataFrame(extreme_data)
  
  return extreme_data

def train_GMMs(conf):
  """
  Build GMM mixture models of training data to use for prediction

  Pickles the resulting models and saves them in conf['pickled_model_path']

  Arguments:
    conf dict Configuration parameters

  Returns:
    bool True when done

  """
  training_data_path = ROOT_PATH / conf['training_data_path']
  training_data = pd.read_csv(str(training_data_path), header=0, dtype={ 'particle_id': str })

  cols = conf['training_cols']

  # Generate extremes of true rupture events
  extremes = training_data.loc[((training_data['true_event'] == 'R') | (training_data['true_event'] == 'E'))].groupby([ 'data_set', 'particle_id', 'true_event_id' ], group_keys=False).apply(get_extremes, cols)

  np.random.seed(12345)
  models = {}
  for col in cols:
    mix = GaussianMixture(n_components=2, covariance_type='full')
    training_y = np.concatenate([
      extremes['value'].loc[( extremes['variable'] == (col+'_min') ) | ( extremes['variable'] == (col+'_max') )], 
      training_data[col].loc[ (training_data['true_event_id'] == -1) ]
    ])
    training_y = training_y[np.isfinite(training_y)]
    training_y = training_y.reshape(-1, 1)
    mix.fit(training_y)

    if mix.converged_ is not True:
      raise RuntimeError(str(col) + " model did not converge")

    # The component with the smaller variance is N
    R_index = np.argsort(mix.covariances_.flatten())[1]

    models[col] = {
      'model': mix,
      'R_index': R_index
    }

  # Save the models
  joblib.dump(models, str(ROOT_PATH / conf['pickled_model_path']))
  return True

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

def seed_events(data, conf):
  """
  Seeds rupture events for full classification

  Arguments:
    p_data pd.DataFrame The particle data
    tiff_path string|Path Path to the TIFFs
    conf dict Various parameters used for finding rupture events

  Returns:
    (bool, pd.DataFrame) Whether seeding completed, The modified particle data
  """

  tmp_path = ROOT_PATH / conf['tmp_dir']
  tmp_path.mkdir(mode=0o755, parents=True, exist_ok=True)
  
  # Load the models
  models = joblib.load(str(ROOT_PATH / conf['pickled_model_path']))

  # Prediction runs as follows:
  # - Models predict the rupture-ness of each frame
  # - The cutoff is [model_rupture_cutoff]: any model that finds a >= 90% probability of belonging to rupture dist 
  #   is scored as a 1, otherwise 0
  # - The mean value of all the scores is calculated
  # - The mean value over time is smoothed with a smooth spline function
  # - If the mean value >= [meta_rupture_cutoff], those frames are considered to be ruptures
  # - To separate rupture events that happen in succession, the 1st derivative of the smoothed spline is calculated
  # - For each zero of the 1st deriv, the start of a rupture is the first frame where the cutoff is found to the zero, 
  #   repair is the zero until the end of the rupture, or until the next zero

  columns = []
  for col,model in models.items():
    mix = model['model']
    R_index = model['R_index']
    y = np.array(data.loc[( pd.notnull(data[col]) ),col]).reshape(-1, 1)

    posterior = mix.predict_proba(y)[...,R_index]
    ruptures = (posterior > conf['model_rupture_cutoff']).astype(int)

    new_col = 'GMM_' + col + '_pred'
    data[new_col] = 0
    data.loc[( pd.notnull(data[col]) ), new_col] = ruptures

    columns.append(new_col)

  data['GMM_prob'] = data[columns].mean(axis=1)
  data = fit_splines(data, [('GMM_prob', 'GMM_prob', 0.1)], tmp_path)
  
  r_idx = ((data['GMM_prob_spline'] >= conf['meta_rupture_cutoff']))
  data.loc[(r_idx), 'event'] = 'R'

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

  if conf['train'] is True:
    print("Training GMMs...")
    train_GMMs(conf)
  
  # Filter out particles that are too near each other
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(
    lambda x: np.min(x['nearest_neighbor_distance']) >= 50*x['x_conversion'].iloc[0]
  )

  # Filter out particles that are too jumpy
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: np.all(x['speed'] <= 0.05)) # Faster than 0.05 µm/3 min

  data.sort_values([ 'data_set', 'particle_id', 'frame' ], inplace=True)
  data.reset_index(inplace=True, drop=True)

  # Classify
  if 'event' not in data.columns:
    data.loc[:,'event'] = 'N'
    data.loc[:,'event_id'] = -1

  data = seed_events(data, conf=conf)

  if not fast:
    try:
      tqdm.pandas(desc="Classifying particles", ncols=90, unit="particle")
      data = data.groupby([ 'data_set', 'particle_id' ]).progress_apply(process_event_seeds, conf=conf)
    except Exception as e:
      return ( False, data )

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

  cp = data.copy()

  # Calculate and store baseline median / particle
  cp = cp.groupby([ 'data_set', 'particle_id' ]).apply(get_baseline_median)

  # Filter out non-events
  cp = cp.loc[( cp['event'] != 'N' ), :]

  if cp.shape[0] <= 0:
    return get_event_info(cp)

  tqdm.pandas(desc="Processing event summaries", ncols=90, unit="event")
  events = cp.groupby([ 'data_set', 'particle_id', 'event_id' ]).progress_apply(get_event_info)
  
  # events = apply_parallel(cp.groupby([ 'data_set', 'particle_id', 'event_id' ]), "Processing event summaries", get_event_info)
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
  event_data = {
    'data_set': [],
    'particle_id': [],
    'event': e_data['event'].unique().tolist(),
    'duration': [],
    'normalized_duration': [],
    'fraction_fp_lost': [],
    'rupture_size': []
  }

  if e_data.shape[0] <= 0:
    return pd.DataFrame(event_data)

  min_stationary_intensity = np.min(e_data['stationary_median'])
  min_intensity = np.min(e_data['median'])
  baseline_intensity = e_data['raw_baseline_median'].unique()[0]

  for event_type in event_data['event']:
    duration = e_data.loc[( e_data['event'] == event_type ), 'event_duration'].unique()[0]
    event_data['duration'].append(duration)

    normalized_duration = duration * abs(min_stationary_intensity)
    event_data['normalized_duration'].append(normalized_duration)

    event_data['fraction_fp_lost'].append(min_intensity/baseline_intensity)

    first_time = np.min(e_data['time'])
    first_velocity = e_data.loc[( e_data['time'] == first_time ), 'median_derivative'].unique()[0]

    # If this is a rupture, the rupture hole can be estimated by the 
    # following linear relationship:
    # area nm = -0.01249x-0.01079 where x is in  s^-1
    event_data['rupture_size'].append(-0.01249*first_velocity-0.01079)

  if len(event_data['duration']) > 1:
    duration_sum = np.sum(event_data['duration'])

    event_data['event'].append("+".join(event_data['event']))
    event_data['duration'].append(duration_sum)
    event_data['normalized_duration'].append(duration_sum * abs(min_stationary_intensity))

    event_data['fraction_fp_lost'].append(min_intensity/baseline_intensity)
    event_data['rupture_size'].append(-0.01249*first_velocity-0.01079)

  event_data['data_set'] = [ e_data['data_set'].unique()[0] ]*len(event_data['event'])
  event_data['particle_id'] = [ e_data['particle_id'].unique()[0] ]*len(event_data['event'])
  
  return pd.DataFrame(event_data)

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
