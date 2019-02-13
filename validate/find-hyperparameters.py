# coding=utf-8

"""
Attempts to find optimal parameters for a given model

Usage:
  find-hyperparameters.py CLASSIFIER [--samples=1000] [--init-samples=10000] [--test-data-file=validation-data/validation-data.csv]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --init-samples=<int> [defaults: 10000] The number of hyperparameter value sets to try on the first round.
  --samples=<int> [defaults: 1000] The number of hyperparameter value sets to try.
  --test-data-file=<string> [defaults: validation-data.csv] The CSV file containing particle data with true events

Output:
  Values for each model hyperparameter
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt
import numpy as np
import pandas as pd
import json
from multiprocessing import Pool, cpu_count
import subprocess
import re
import time
import math

### Constants
VALIDATE_PATH     = (ROOT_PATH / ("validate/validate.py")).resolve()
CONVERGENCE_LIMIT = 0.5 # How close scores have to be before we stop
MAX_PROCESSES     = 4

### Arguments and inputs
arguments = docopt(__doc__, version='NE-classifier 0.1')

num_samples = int(arguments['--samples']) if arguments['--samples'] else 1000
init_num_samples = int(arguments['--init-samples']) if arguments['--init-samples'] else 2000

classifier = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier))

data_file_path = Path(arguments['--test-data-file']).resolve() if arguments['--test-data-file'] else Path("validate/validation-data/validation-data.csv").resolve()
if not data_file_path.exists():
  print(colorize("red", "Data file input cannot be found: \033[1m" + str(data_file_path) + "\033[0m"))
  exit(1)

hyper_conf_path = (ROOT_PATH / ("classifiers/" + classifier + "/hyperparameters.json")).resolve()
conf_path = (ROOT_PATH / ("classifiers/" + classifier + "/conf.json")).resolve()

def run_epoch(validate_path, hyper_conf_path, num_samples, save_path=None):
  """
  Runs an epoch of [num_samples]

  Uses random search to try and identify the optimal parameters for the 
  given classifier. Will use the JSON file in [hyper_conf_path] to sample 
  [num_samples] and test all of them, using multi-threading.

  Once all have been tested, will overwrite [hyper_conf_path], using the 
  best scoring parameter as the mean, and the SD of the top 10% as the 
  new SD, if applicable.

  Returns the highest-performing score.

  Arguments:
    validate_path PathLib path The full path to the validate.py script
    hyper_conf_path PathLib path The full path to the hyeprparameters.json file
    num_samples int The number of parameters to sample
    save_path PathLib path|None The full path to where the results should be saved

  Returns:
    float The top score among all [num_samples] samples.
  """

  start_time = time.time()

  # Get hyper_conf
  with hyper_conf_path.open(mode='r') as file:
    hyper_conf = json.load(file)

  # Get conf
  with conf_path.open(mode='r') as file:
    conf = json.load(file)

  # Sample possible values for each parameter in [hyper_conf]
  column_map = {}
  values = np.zeros(( num_samples, len(hyper_conf)+2 ), dtype=np.float64)

  idx = 0
  for name, param in hyper_conf.items():
    if param['type'] == "gaussian":
      values[:,idx] = np.random.normal(param['mean'], param['sd'], num_samples)

    elif param['type'] == "beta":
      values[:,idx] = np.random.beta(param['a'], param['b'], num_samples)

    elif param['type'] == "uniform":
      values[:,idx] = np.random.uniform(param['min'], param['max']+0.001, num_samples)

    else:
      continue

    # This will tell us which column in our matrix corresponds to what param
    column_map[name] = ( idx, param['dtype'] )
    idx = idx + 1

  mean_col = idx
  sd_col = idx+1
  wheel_idx = 0
  num_failures = 0

  # Run each sample set
  results = np.zeros(( num_samples, 2), dtype=np.float64)
  for i in range(0,num_samples):
    wheel_idx = print_progress_bar(num_samples, i, num_failures, wheel_idx)
    result = run_sample(validate_path, conf, column_map, values[i])
    if result is False:
      num_failures = num_failures+1
      results[i] = [ 0.0, 0.0 ]
    else:
      results[i] = result

  print_progress_bar(num_samples, num_samples, num_failures, wheel_idx)

  values[:,mean_col] = results[:,0]
  values[:,sd_col] = results[:,1]

  # Find the highest scoring params
  values = values[values[:,mean_col].argsort()]
  values = np.flip(values,0)
  to_keep = math.ceil(num_samples/2)

  # Build new hyper conf
  for name, param in hyper_conf.items():
    column_info = column_map[name]
    column_idx = column_info[0]
    dtype = column_info[1]

    if param['type'] == "gaussian":
      mean = values[0,column_idx]
      sd = np.std(values[0:to_keep,column_idx])*1.5 # Make this a little bigger so we don't get trapped in a local maxima
      if dtype == 'int':
        mean = int(mean)
        sd = int(sd)
      hyper_conf[name]['mean'] = mean
      hyper_conf[name]['sd'] = sd

    elif param['type'] == "beta":
      mean = np.mean(values[0:to_keep,column_idx])
      if dtype == 'int':
        mean = int(mean)
      hyper_conf[name]['a'] = (100*mean)-5
      hyper_conf[name]['b'] = 5

    elif param['type'] == "uniform":
      min_val = np.min(values[0:to_keep, column_idx])
      max_val = np.max(values[0:to_keep, column_idx])
      if dtype == 'int':
        min_val = int(min_val)
        max_val = int(max_val)
      hyper_conf[name]['min'] = min_val
      hyper_conf[name]['max'] = max_val

  # Write out new hyper conf file
  with hyper_conf_path.open(mode='w') as file:
    file.write(json.dumps(hyper_conf))

  # Write out results
  if save_path is not None:
    np.savetxt(str(save_path), values, delimiter=",")

  top_score = values[0,mean_col]
  run_time = time.time() - start_time

  print(f"  Top score: {top_score}")
  print("  Run in : {:2.2f}".format(run_time))
  return top_score

def run_sample(validate_path, conf, column_map, values):
  """
  Tests a particular sample set

  Calls [validate_path] with the supplied values

  Arguments:
    validate_path PathLib path The complete path to the validate.py program
    conf dict The existing conf object
    column_map dict Metadata about our values
    values list The values to use for our classifier

  Returns:
    list The mean score and SD
  """
  for param, idx in column_map.items():
    conf[param] = values[idx[0]]
    if idx[1] == "int":
      conf[param] = int(conf[param])

  cmd = [
    "python",
    str(validate_path),
    classifier,
    "--classifier-conf=" + json.dumps(conf),
    "--simple-output=1",
    "--max-processes=8"
  ]

  try:
    out = subprocess.check_output(cmd)
  except:
    return False

  return out.split()

def print_progress_bar(total, num_completed, num_failures, wheel_idx):
  """
  Prints a friendly progress bar

  Arguments:
    total int The total number of processes to run
    num_completed int The number of processes finished
    wheel_idx int Used for printing a porgress wheel
    num_failures int The number of processes that failed

  Returns:
    int The wheel_idx to use next
  """
  wheels = [
    "|",
    "/",
    "-",
    "\\",
    "|",
    "/",
    "-",
    "\\"
  ]

  if wheel_idx >= len(wheels):
    wheel_idx = 0

  human_count = num_completed+1 if num_completed != total else num_completed # Just for the loading bar
  progress = int(30*num_completed//total)
  bar = "#" * progress + ' ' * (30 - progress)
  wheel = wheels[wheel_idx]
  failure_text = ""
  print(f"\r  \033[1mScoring parameters \033[0m |{bar}| ({human_count}/{total}, {num_failures} failed) {wheel}", end="\r")

  if num_completed == total:
    print()

  return wheel_idx+1


old_score = 0
current_score = abs(CONVERGENCE_LIMIT)+1
i = 1
while(abs(current_score-old_score) > CONVERGENCE_LIMIT):
  print(f"Running epoch {i}:")
  old_score = current_score
  save_path = (ROOT_PATH / ("classifiers/" + classifier + "/epoch-" + str(i) + ".csv"))
  if i == 1:
    current_score = run_epoch(VALIDATE_PATH, hyper_conf_path, init_num_samples, str(save_path))
  else:
    current_score = run_epoch(VALIDATE_PATH, hyper_conf_path, num_samples, str(save_path))
  i = i+1
