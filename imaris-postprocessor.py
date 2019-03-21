# coding=utf-8

"""Imaris post processing

Takes Imaris output files and generates a combined dataframe with normalized and scaled values.

Usage:
  imaris-postprocessor.py INPUT_DIR OUTPUT_DIR [--frame-rate=180] [--data-set=GID]

Arguments:
  INPUT_DIR Path to Imaris data sheets
  OUTPUT_DIR Path to output predictions and videos (if selected)

Options:
  --data_set=<string|0> The universally unique ID to identify this data set. If falsey, defaults to generated UUID.
  --img-dir=<string> [defaults: INPUT/../images] The directory that contains TIFF images of each frame, for outputting videos.
  --frame-rate=<int> [defaults: 180] The seconds that elapse between frames

Output:
  Writes a CSV to OUTPUT_DIR with post-processed data and JSON file with meta-data
"""
import sys
import os

from common.docopt import docopt
from common.version import get_version

import numpy as np
import pandas as pd
import csv
from pathlib import Path
import uuid
from errors.UnexpectedEOFException import UnexpectedEOFException
from scipy import interpolate
from statsmodels.tsa.stattools import kpss
from PIL import Image

"""
Arguments and inputs
"""
arguments = docopt(__doc__, version=get_version())

input_path = Path(arguments['INPUT_DIR']).resolve()
output_path = Path(arguments['OUTPUT_DIR']).resolve()
tiff_path = input_dir / (arguments['--img-dir']) if arguments['--img-dir'] else (input_dir / ("../images/")).resolve()
frame_rate = int(arguments['--frame-rate']) if arguments['--frame-rate'] else 180
gid = arguments['--data-set'] if arguments['--data-set'] else str(uuid.uuid4())

def colorize(color, string):
  """
  Used to print colored messages to terminal

  Arguments:
    color string The color to print
    string string The message to print

  Returns:
    A formatted string
  """
  colors = {
    "red": "31",
    "green": "32",
    "yellow": "33", 
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37"
  }

  return "\033[" + colors[color] + "m" + string + "\033[0m"

""""
What Imaris CSV files we need and their columns
"""
data_files = {
  'position': { 
    'file': 'Position.csv', 
    'data_cols': [ 'TrackID', 'Time', 'Position X', 'Position Y' ] 
  },
  'area': { 
    'file': 'Area.csv', 
    'data_cols': [ 'TrackID', 'Time', 'Area' ] 
  },
  'sum': { 
    'file': 'Intensity_Sum_Ch=1.csv', 
    'data_cols': [ 'TrackID', 'Time', 'Intensity Sum' ] 
  },
  'median': { 
    'file': 'Intensity_Median_Ch=1.csv', 
    'data_cols': [ 'TrackID', 'Time', 'Intensity Median' ] 
  },
  'oblate_ellipticity': {
    'file': 'Ellipticity_Oblate.csv',
    'data_cols': [ 'TrackID', 'Time', 'Ellipticity (oblate)' ]
  }
}

"""
Constant for mappng the Imaris column names to easier ones to type
"""
COL_MAP = {
  'TrackID': 'particle_id',
  'Time' : 'frame',
  'Position X' : 'x',
  'Position Y' : 'y',
  'Area': 'area',
  'Intensity Median': 'median',
  'Intensity Sum': 'sum',
  'Ellipticity (oblate)': 'oblate_ellipticity'
}


### Collect our data
data = None

for name,csv_info in data_files.items():
  file_path = input_path / csv_info['file']
  
  if not file_path.exists():
    print(colorize("yellow", str(file_path) + " not found; skipping"))
    continue

  with open(file_path, 'rb') as f:
    # Find the actual start of CSV data
    # Imaris outpus really silly headers
    found_header = False
    for line in f:
      line = line.decode('utf-8').rstrip()
      if line.find(",") != -1:
        # Extract header
        headers = line.split(",")
        if(len(list(set(headers) & set(csv_info['data_cols']))) > 0):
          found_header = True
          break

    if not found_header:
      raise UnexpectedEOFException("Reached end of " + csv_info['file'] + " without finding headers.")

    this_df = pd.read_csv(f, header=None, names=headers)
    this_df = this_df[csv_info['data_cols']]
    if(data is None):
      data = this_df
    else:
      data = pd.merge(data, this_df, how='left', on=[ 'TrackID', 'Time' ])

data.rename(columns=COL_MAP, inplace=True)
data['particle_id'] = data['particle_id'].astype(str)
data['time'] = data['frame']*frame_rate

# Add data set ID and seconds/frame
data['data_set'] = gid
data['frame_rate'] = frame_rate

# Sort data
data = data.sort_values(by=[ 'data_set', 'particle_id', 'time' ])

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

### Filter out short tracks
data = data.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: x['frame_rate'].iloc[0]*len(x) > 28800)


### Normalize median intensity by average particle intensity/frame
def normalize_intensity(group):
  """
  Normalize median intensity by average particle intensity/frame

  Arguments:
    group Pandas DataFrame of each frame

  Returns:
    Modified Pandas DataFrame
  """
  mean_median = np.mean(group['median'])
  group['normalized_median'] = group['median']/mean_median
  group['normalized_sum'] = group['sum']/mean_median
  return group

data = data.groupby([ 'frame' ]).apply(normalize_intensity)
data = scale(data, 'normalized_median', 'normalized_median')
data = scale(data, 'normalized_sum', 'normalized_sum')


### Scale area of each particle to be between 0 and 1 (relative to itself)
data = data.groupby([ 'data_set', 'particle_id' ]).apply(scale, 'area', 'scaled_area')


### Make intensity/sum/area stationary
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

data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'scaled_area', 'stationary_area')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'normalized_median', 'stationary_median')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(make_stationary, 'normalized_sum', 'stationary_sum')


### Interpolate with cubic splines/find derivatives
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

data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'scaled_area', 'area')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_median', 'median')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'normalized_sum', 'sum')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'x', 'x')
data = data.groupby([ 'data_set', 'particle_id' ]).apply(fit_spline, 'y', 'y')


### Make particle IDs less… long. This assumes we won't have more than 999 particles in a video.
data['particle_id'] = data['particle_id'].str.replace("1000000", "")


### Get pixel to micron conversion factor
frame_file_name = str(np.min(data['frame'])).zfill(4) + '.tif'
frame_path = (tiff_path / (gid + "/" + frame_file_name)).resolve()
with Image.open(frame_path) as img:
  resolution = img.info['resolution']
  x_conversion = resolution[0]
  y_conversion = resolution[1]

data['x_conversion'] = x_conversion
data['y_conversion'] = y_conversion


### Write out the files
output_path.mkdir(mode=0o755, parents=True, exist_ok=True)

csv_path = output_path / "data.csv"

data.to_csv(str(csv_path), header=True, encoding='utf-8', index=None)