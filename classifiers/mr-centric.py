# coding=utf-8

"""
NE rupture classifier

Uses Python 3 and input from Imaris to identify cells that have undergone NE rupture and repair.
Attempts to distinguish: rupture, repair, and mitosis

Usage:
  mr-centric.py DATA_FILE OUTPUT [--file-name=results.csv] [--run-qa=0] [--img-dir=0] [--conf=0] [--max-processes=None]

Arguments:
  DATA_FILE string Path to processed data csv file
  OUTPUT string Path to where the classified data CSV file should be saved

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
  In addition, if tiff-dir is supplied, annotated videos of each cell-event will be produced, 
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
from multiprocessing import Pool, cpu_count
import subprocess
from time import sleep

arguments = docopt(__doc__, version='NE-classifier 0.1')

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()
CONF_PATH = (ROOT_PATH / ("classifiers/mr-centric/conf.json")).resolve()
if arguments['--conf'] is not "0":
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

  - Seeds mitosis/rupture events (indistinguishable at this point)
  - Find the end of the event (when intensity returns to baseline)
  - Disambiguate ruptures from mitoses by how long it takes to recover
  - For ruptures, find when repair begins

  Arguments:
    p_data Panda DataFrame The particle data
    convergence_limit float Convergence limit; see description 

  Returns:
    Panda DataFrame
  """

  # Seed events
  p_data.loc[:,'event'] = 'N'
  p_data.loc[:,'event_id'] = -1
  p_data = seed_events(p_data, CONF, convergence_limit)
  idx = (
    p_data['event_id'] != -1
  )
  
  # Find event ends
  p_data = extend_events(p_data, CONF, convergence_limit)
  idx = (
    p_data['event_id'] != -1
  )

  # Find when events begin to recover
  p_data = find_event_recoveries(p_data, CONF)

  # Disambiguate events
  p_data = disambiguate_events(p_data, CONF)

  # Clean up
  p_data = p_data.groupby(['event_id']).apply(cleanup)

  return p_data

def seed_events(p_data, conf, convergence_limit):
  """
  Seeds mitosis/rupture events for full classification
  
  Filters out events where the event and its following frame
  are above the baseline intensity (mean intensity of non-event frames). 
  Since filtering will change the baseline intensity, we continue 
  iterating until the squared difference between the new and old 
  baseline intensities is less than convergence_limit.

  Arguments:
    p_data Panda DataFrame The particle data
    conf list Cutoffs for various parameters used for finding events
    convergence_limit float When to stop filtering

  Returns:
    Panda DataFrame The modified particle data
  """

  col_names = list(p_data.columns.values)

  baseline_median = 1000.0
  old_baseline_median = 0.0

  # Since removing events will change baseline, we will cycle
  # until we converge at a stable set of event seeds
  while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
    p_data.loc[:,'next_median'] = p_data['normalized_median'].shift(-1)

    idx = (
      ((p_data['normalized_median'] <= baseline_median*conf['baseline_median_scale_factor']) | (p_data['next_median'] <= baseline_median*conf['baseline_median_scale_factor'])) &
      (p_data['median_derivative'] <= conf['median_derivative']) &
      (p_data['area_derivative'] >= conf['area_derivative']) 
    )

    p_data.loc[:,'event'] = 'N'
    p_data.loc[idx, 'event'] = 'MR'
    old_baseline_median = baseline_median
    baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'normalized_median'])

  p_data.loc[idx, 'event_id'] = (p_data.loc[idx,'frame'].diff() > 1).cumsum()
  
  return p_data
  # return p_data.loc[:, col_names]

def extend_events(p_data, conf, convergence_limit):
  """
  Extend seeded events

  For each event, will extend the event out till 
  baseline intensity is reached. Since this will
  in turn affect the baseline intensity, this is
  iterated for each event till convergence is reached.

  Arguments:
    p_data Panda DataFrame The particle data
    convergence_limit float When to stop filtering

  Returns
    Panda DataFrame The modified data frame
  """
  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    old_baseline_median = 0.0

    no_event_idx = (p_data['event'] == 'N')
    if no_event_idx.any():
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'stationary_median'])
    else:
      baseline_median = 0

    while(math.pow(baseline_median - old_baseline_median, 2) >= convergence_limit):
      start = np.max(p_data.loc[(p_data['event_id'] == event_id), 'frame'])

      reach_baseline_idx = (
        (p_data['frame'] > start) &
        (p_data['normalized_median'] > baseline_median*conf['baseline_median_scale_factor'])
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

      if next_start_idx.any():
        # If this isn't the last event in the series, find the start of the
        # next event
        next_start = np.min(p_data.loc[next_start_idx,'frame'])
      else:
        next_start = np.max(p_data['frame'])+1

      stop = np.min([ reach_baseline, next_start ])

      idx = (
        (p_data['frame'] > start) &
        (p_data['frame'] < stop)
      )
      p_data.loc[idx, 'event'] = 'MR'
      p_data.loc[idx, 'event_id'] = event_id

      old_baseline_median = baseline_median
      baseline_median = np.mean(p_data.loc[(p_data['event'] == 'N'), 'normalized_median'])

    # Extend backwards 1 frame
    idx = (
      (p_data['frame'] == start-1) & 
      (p_data['event_id'] == event_id)
    )
    p_data.loc[idx, 'event'] = 'MR'
    p_data.loc[idx, 'event_id'] = event_id
  
  return p_data

def find_event_recoveries(p_data, conf):
  col_names = list(p_data.columns.values)

  # Get a smoothed median intensity
  frame_rate = p_data['frame_rate'].unique()[0]
  p_data.loc[:, 'smoothed'] = sliding_average(p_data['normalized_median'], conf['sliding_window_width'], conf['sliding_window_step'], frame_rate)

  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    # Find where we start returning to baseline, preferentially using the
    # smoothed value, but the normalized value if we have to

    event = p_data.loc[(p_data['event_id'] == event_id),:]
    
    idx = (event['smoothed'].diff() > 0)

    if not idx.any():
      idx = (event.loc[:, 'normalized_median'].diff()) > 0

    if idx.any():
      recovery_start = np.min(event.loc[idx, 'time'])
      p_data.loc[((p_data['time'] >= recovery_start) & (p_data['event_id'] == event_id)), 'event'] = "MRR"

  return p_data.loc[:, col_names]

def disambiguate_events(p_data, conf):
  event_ids = p_data.loc[(p_data['event_id'] != -1), 'event_id'].unique()

  for event_id in event_ids:
    # How long did it take to start recovering?
    event_idx = (p_data['event_id'] == event_id)
    times = p_data.loc[event_idx, 'time']

    idx = (
      (p_data['event_id'] == event_id) &
      (p_data['event'] == 'MRR')
    )

    event_start = np.min(times)

    if not idx.any():
      # We never started recovering
      recovery_start = np.min(times)
    else:
      recovery_start = np.min(p_data.loc[idx, 'time'])
    
    diff = recovery_start - event_start

    if diff > conf['max_rupture_delay']:
      # This is mitosis
      p_data.loc[event_idx, 'event'] = 'M'
    else:
      p_data.loc[event_idx, 'event'] = 'R'
      p_data.loc[((p_data['event_id'] == event_id) & (p_data['time'] >= recovery_start)), 'event'] = 'E'

  return p_data

def sliding_average(data, window, step, frame_rate):
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

def cleanup(group):
  if len(group['event'].unique()) == 1 and group['event'].unique()[0] == "E":
    group['event'] = 'N'
    group['event_id'] = -1

  return group

def print_progress_bar(total, num_left):
  """
  Prints a nice progress bar onto the terminal

  Arguments:
    total int The number of child processes to run
    num_left int The number of child processes left to run
  """
  progress = int(30*(total-num_left)//total)
  bar = "#" * progress + ' ' * (30 - progress)
  print("\r\033[1mClassifying particles \033[0m|%s|" % bar, end="\r")

  if num_left == 0:
    print()

def apply_parallel(max_processes, grouped, fn, *args):
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

  with Pool(max_processes) as p:
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
  return pd.concat(rs.get())

if __name__ == '__main__':
  # chunk = classify_particle_events(data.loc[(data['particle_id'] == '034'),:].copy())
  # exit()
  data = apply_parallel(max_processes, data.groupby([ 'data_set', 'particle_id' ]), classify_particle_events)
  
output_path.mkdir(exist_ok=True)

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

