# coding=utf-8

"""
NE rupture classifier

Uses Python 3 and input from Imaris to identify cells that have undergone NE rupture and repair.
Attempts to distinguish: rupture, repair, and mitosis/death (the latter are not yet distinguishable)
Also attempts to identify the rate of GFP import during repair by fitting a one-phase exponential curve 
to the intensity data. (Not yet.)

Usage:
  basic.py DATA_FILE OUTPUT [--file-name=results.csv] [--run-qa=0] [--img-dir=0] [--conf=0] [--max-processes=None]

Arguments:
  DATA_FILE Path to processed data csv file
  OUTPUT Path to where the classified data CSV file should be saved

Options:
  -h --help Show this screen.
  --version Show version.
  --file-name=<string> [defaults: results.csv] The name of the resulting CSV file
  --run-qa=<bool> [defaults: False] If True, will output graphs and videos
  --img-dir=<string> [defaults: None] The directory that contains TIFF images of each frame, for outputting videos.
  --conf=<string> [defaults: None] Override configuration options in conf.json with a JSON string.
  --max-processes=<int> [defaults: cpu_count()] The number of processes this classifier can use

Output:
  Writes a CSV to DATA_DIR with all classified events (the null event is not included).
  In addition, if run-qa is true and img-dir is supplied, annotated videos of each cell-event will be produced, 
  as well as a full-size, annotated video.
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt

import math
import numpy as np
import pandas as pd
import csv
import json
from scipy.stats import iqr
from multiprocessing import Pool, cpu_count
import subprocess
from time import sleep

arguments = docopt(__doc__, version='NE-classifier 1.0')

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()
CONF_PATH = (ROOT_PATH / ("classifiers/outliers/conf.json")).resolve()
if arguments['--conf'] is not None and arguments['--conf'] is not "0":
  CONF = json.loads(arguments['--conf'])
else:
  with CONF_PATH.open(mode='r') as file:
    CONF = json.load(file)

### Arguments and inputs
data_file_path = (ROOT_PATH / (arguments['DATA_FILE'])).resolve()
output_path = (ROOT_PATH / (arguments['OUTPUT'])).resolve()
file_name = arguments['--file-name'] if arguments['--file-name'] else "results.csv"
run_qa = True if arguments['--run-qa'] else False
tiff_path = Path(arguments['--img-dir']).resolve() if arguments['--img-dir'] else ""
max_processes = int(arguments['--max-processes']) if arguments['--max-processes'] else cpu_count()


### Read our data
data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

def classify_particle_events(p_data, convergence_limit = 3E-6):
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

  # Seed events
  p_data.loc[:,'event'] = 'N'
  p_data.loc[:,'event_id'] = -1
  p_data = seed_events(p_data, CONF)

  # Give an ID to each event
  idx = (p_data['event'] != 'N')
  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()

  # Find event ends
  p_data = extend_events(p_data, CONF)

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, CONF)

  # Figure out what the events are
  #p_data = classify_events(p_data, CONF)

  # Filter out events that were captured incorrectly
  #p_data = filter_events(p_data, CONF)

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

  idx = (
    (p_data['stationary_median'] < np.quantile(p_data['stationary_median'], 0.25)-conf['median_cutoff']*iqr(p_data['stationary_median'])) &
    (p_data['stationary_area'] < np.quantile(p_data['stationary_area'], 0.25)+conf['area_cutoff']*iqr(p_data['stationary_area']))
  )

  idx = get_initial_ruptures(p_data, conf)
  p_data.loc[idx, 'event'] = '?'

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

    while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
      start = np.max(p_data.loc[(p_data['event_id'] == event_id), 'frame'])

      reach_baseline_idx = (
        (p_data['frame'] > start) &
        (p_data['stationary_median'] > baseline_median*conf['baseline_scale_factor'])
      )

      if not reach_baseline_idx.any():
        # There are no frames beyond start that reach baseline
        reach_baseline = np.max(p_data['frame'])+1
      else:
        # Get the first frame that satisfies the above conditions
        reach_baseline = np.min(p_data.loc[reach_baseline_idx, 'frame'])

      # Find the next event
      next_start_idx = (
        (p_data['event_id'] != -1) & 
        (p_data['event_id'] != event_id) & 
        (p_data['frame'] > start)
      )

      idx = (
        (p_data['frame'] > start) &
        (p_data['frame'] < reach_baseline)
      )
      p_data.loc[idx, 'event'] = '?'
      p_data.loc[idx, 'event_id'] = event_id

      old_baseline_median = baseline_median
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])

    # Extend backwards 1 frame
    idx = (
      (p_data['frame'] == start-1) & 
      (p_data['event_id'] == event_id)
    )
    p_data.loc[idx, 'event'] = '?'
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
  p_data.loc[:, 'smoothed'] = sliding_average(p_data['stationary_median'], conf['sliding_window_width'], conf['sliding_window_step'], frame_rate)

  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    # Find where we start returning to baseline, preferentially using the
    # smoothed value, but the normalized value if we have to

    event = p_data.loc[(p_data['event_id'] == event_id),:]
    event_type = event['event'].iloc[0]
    recovery_label = 'R'
    
    idx = (event['smoothed'].diff() > 0)

    if not idx.any():
      idx = (event.loc[:, 'stationary_median'].diff()) > 0

    if idx.any():
      recovery_start = np.min(event.loc[idx, 'time'])
      event_start = np.min(event.loc[:,'time'])
      p_data.loc[((p_data['time'] >= recovery_start) & (p_data['time'] != event_start) & (p_data['event_id'] == event_id)), 'event'] = recovery_label

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

if __name__ == '__main__':
  # groups = data.groupby([ 'data_set', 'particle_id' ])
  # results = []
  # for name, group in groups:
  #   if name[1] != '053':
  #     continue
  #   res = classify_particle_events(group)
  #   results.append(res)
  # exit()

  # data = pd.concat(results)
  data = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), classify_particle_events)

output_file_path = (output_path / (file_name)).resolve()
data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

if run_qa:
  cmd = [
    "python",
    str(QA_PATH),
    os.path.splitext(os.path.basename(__file__))[0],
    output_file_path,
    str(output_path),
    "--img-dir=" + str(tiff_path)
  ]
  subprocess.call(cmd)

