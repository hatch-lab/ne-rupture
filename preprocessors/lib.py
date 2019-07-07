import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import spatial
from statsmodels.tsa.stattools import kpss

def base_transform(data, params):
  data_set = params['data_set']
  frame_rate = params['frame_rate']

  data['frame_rate'] = frame_rate
  data['time'] = data['frame']*frame_rate
  data['data_set'] = data_set

  # Filter short tracks
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: x['frame_rate'].iloc[0]*len(x) > 28800)

  # Sort data
  data = data.sort_values(by=[ 'data_set', 'particle_id', 'time' ])

  # Normalize median intensity by average particle intensity/frame
  data = data.groupby([ 'frame' ]).apply(normalize_intensity, 'median', 'median', 'normalized_median')
  data = data.groupby([ 'frame' ]).apply(normalize_intensity, 'median', 'sum', 'normalized_sum')

  data = scale(data, 'normalized_median', 'normalized_median')
  data = scale(data, 'normalized_sum', 'normalized_sum')

  # Scale area of each particle to be between 0 and 1 (relative to itself)
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(scale, 'area', 'scaled_area')

  # Make intensity/sum/area stationary
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'scaled_area', 'stationary_area')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'normalized_median', 'stationary_median')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'normalized_sum', 'stationary_sum')

  data = data.groupby([ 'data_set', 'particle_id' ]).apply(z_score, 'mean_cyto', 'z_cyto')

  # Interpolate with cubic splines/find derivatives
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'scaled_area', 'area')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_median', 'median')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_sum', 'sum')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'x', 'x')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'y', 'y')

  # Find nearest neighbors
  data = data.groupby([ 'data_set', 'frame' ]).apply(find_nearest_neighbor_distances)

  return data

def scale(group, col, new_col):
  """
  Scales a column to be between 0 and 1

  Arguments:
    group Pandas DataFrame of each particle
    col string The column to scale
    new_col string The new column to store scaled values

  Returns:
    DataFrame Modified Pandas DataFrame
  """

  max_val = np.max(group[col])
  min_val = np.min(group[col])

  group[new_col] = (group[col]-min_val)/(max_val-min_val)
  
  return group

def sliding_average(data, window, step, frame_rate):
  """
  Calculates the sliding average of a column

  Arguments:
    data list The data to operate on
    window int The size of the window in seconds
    step int How far to advance the window in seconds
    frame_rate int The number of seconds in a frame

  Returns:
    list The sliding window average of equal length to data
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

def normalize_intensity(group, normalize_by, col, new_col):
  """
  Normalize median intensity by average particle intensity/frame

  Arguments:
    group Pandas DataFrame of each frame

  Returns:
    Modified Pandas DataFrame
  """
  mean_val = np.mean(group[normalize_by])
  group[new_col] = group[col]/mean_val
  return group

def z_score(group, col, new_col):
  """
  Find z-score of given column

  Arguments:
    group Pandas DataFrame of each frame

  Returns:
    Modified Pandas DataFrame
  """
  mean_val = np.mean(group[col])
  sd = np.std(group[col])
  group[new_col] = (group[col]-mean_val)/sd
  return group

def make_stationary(group, col, new_col):
  """
  If a [col] is not stationary, apply 1st order difference

  Arguments:
    group Pandas DataFrame of each particle
    col string The column to test

  Returns:
    Modified Pandas DataFrame
  """
  import warnings
  warnings.simplefilter("ignore")
  
  result = kpss(group[col], regression='c')
  test_stat = result[0]
  critical_val = result[3]['1%']

  if test_stat <= critical_val:
    # Stationary
    group[new_col] = group[col]
  else:
    group = group.sort_values(by=["time"])
    frame_rate = group['frame_rate'].iloc[0]
    smoothed_mean = sliding_average(group[col], 3600, 1800, frame_rate)
    group[new_col] = group[col] - smoothed_mean
    group.loc[(group[new_col] == group[col]), new_col] = np.nan
    
  # Move mean value to 0
  group[new_col] = group[new_col] - np.mean(group[new_col])

  return group

def fit_spline(group, fit_column, new_column_stem):
  """
  Fits cubic splines and calc's derivatives

  Arguments:
    group Pandas DataFrame of each particle
    fit_column string The column name to fit
    new_column_stem string The prefix of the new columns

  Returns:
    Modified Pandas DataFrame
  """
  fit = interpolate.splrep(group['time'], group[fit_column])

  spline_column = new_column_stem + '_spline'
  deriv_column = new_column_stem + '_derivative'

  group[spline_column] = interpolate.splev(group['time'], fit)
  
  # Find derivative
  group[deriv_column] = interpolate.splev(group['time'], fit, der=1)

  return group

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