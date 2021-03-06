import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import spatial

def base_transform(data, params):
  """
  Generate information we always want

  Filters out tracks < 8 hours
  Generates normalized median values (median / avg median per frame)
  Generates normalized sum values (sum / avg median per frame)
  Scales area to be between 0-1 for each cell
  Generates stationary normalized median, normalized sum, scaled and area values
  Generates a z-score for cytoplasmic intensity
  Finds velocities for scaled area, normalized median, normalized sum, x, and y

  Arguments:
    data pandas.DataFrame The existing data
    params dict A dictionary of preprocessor params

  Return:
    pandas.DataFrame The modified data frame
  """
  data_set = params['data_set']
  frame_rate = params['frame_rate']
  frame_width = params['frame_width']
  frame_height = params['frame_height']

  data['frame_rate'] = frame_rate
  data['time'] = data['frame']*frame_rate
  data['data_set'] = data_set

  # Filter short tracks
  if np.max(data['frame'])*frame_rate > 28800:
    data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: x['frame_rate'].iloc[0]*len(x) > 28800)
  else:
    # Need at least 4 datapoints fo fit splines
    data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: len(x) > 3)

  # Filter out particles that are too near the edge
  data = data.groupby([ 'data_set', 'particle_id' ]).filter(
    lambda x, frame_width, frame_height: np.min(x['x_px']) >= 50 and np.max(x['x_px']) <= frame_width-50 and np.min(x['y_px']) >= 50 and np.max(x['y_px']) <= frame_width-50, 
    frame_width = frame_width, frame_height=frame_height
  )

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

  # Interpolate with cubic splines/find derivatives
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'scaled_area', 'area')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_median', 'median')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_sum', 'sum')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'x', 'x')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'y', 'y')
  data = data.groupby([ 'data_set', 'particle_id' ]).apply(find_speed)

  # Find nearest neighbors
  data = data.groupby([ 'data_set', 'frame' ]).apply(find_nearest_neighbor_distances)

  data = data.astype({ 'frame': np.uint64 })

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

def make_stationary(group, col, new_col):
  """
  If a [col] is not stationary, apply 1st order difference

  Arguments:
    group Pandas DataFrame of each particle
    col string The column to test

  Returns:
    Modified Pandas DataFrame
  """
  # group = group.sort_values(by=["time"])
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

def find_speed(group):
  """
  Finds the total speed (magnitude of velocity) given component velocities

  Arguments:
    group Pandas DataFrame of each particle

  Returns:
    Modified Pandas DataFrame
  """
  group['speed'] = np.sqrt(np.add(np.power(group['x_derivative'], 2), np.power(group['y_derivative'], 2)))

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
  if len(idxs) < 2:
    f_data['nearest_neighbor'] = ""
    f_data['nearest_neighbor_distance'] = 0.0
    return f_data

  neighbor_ids = f_data['particle_id'].iloc[idxs].tolist()
  f_data['nearest_neighbor'] = neighbor_ids
  f_data['nearest_neighbor_distance'] = distances

  return f_data