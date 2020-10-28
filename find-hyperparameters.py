# coding=utf-8

"""
Attempts to find optimal parameters for a given model

Usage:
  find-hyperparameters.py CLASSIFIER [--test-data-folder=validate/validation-data/input/] [--steps=3600] [--skip-graphs=0]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --test-data-folder=<string> [defaults: validate/validation-data/input] The directory with the CSV file containing particle data with true events
  --steps=<int> [defaults: 3600] The number of values to try for each hyper parameter
  --skip-graphs=<bool> [defaults: False] If true, will not print AUC curves

Output:
  CSV file with TP, and FN rates for drawing AUC
"""
import sys
import os
from pathlib import Path
from importlib import import_module

ROOT_PATH = Path(__file__ + "/..").resolve()
sys.path.append(str(ROOT_PATH))


from common.docopt import docopt
from common.version import get_version
from common.output import colorize

from validate.lib import get_cell_stats,get_summary_table

from collections import deque
import numpy as np
import pandas as pd
import json
import subprocess
import re
import time
import math

### Constants
R_GRAPH_PATH = (ROOT_PATH / ("validate/find-hyperparameters/make-graphs.R")).resolve()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

classifier_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier_name != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier_name))

data_file_path = Path(arguments['--test-data-folder']).resolve() if arguments['--test-data-folder'] else Path("validate/validation-data/input/data.csv").resolve()
if not data_file_path.exists():
  print(colorize("red", "Data folder input cannot be found: \033[1m" + str(data_file_path) + "\033[0m"))
  exit(1)

steps = int(arguments['--steps']) if arguments['--steps'] else 3600
skip_graphs = bool(arguments['--skip-graphs']) if arguments['--skip-graphs'] else False

hyper_conf_path = (ROOT_PATH / ("classifiers/" + classifier_name + "/hyperparameters.conf.json")).resolve()
conf_path       = (ROOT_PATH / ("classifiers/" + classifier_name + "/conf.json")).resolve()
output_path     = (ROOT_PATH / ("validate/output/" + classifier_name)).resolve()

### Get our classifier
if not (ROOT_PATH / ("classifiers/" + classifier_name + ".py")).exists():
  print(colorize("red", "No such classifier exists"))

classifier = import_module("classifiers." + classifier_name)

data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

def run_model(classifier, conf, data):
  """
  
  """
  classified_data = classifier.run(data, tiff_path=None, conf=conf, fast=True)[1]

  # Get summary data
  results = classified_data.groupby([ 'data_set', 'particle_id' ]).apply(get_cell_stats, False)
  summary = []
  summary.append(get_summary_table(results, "All"))
  for data_set in results['data_set'].unique():
    summary.append(get_summary_table(results[(results['data_set'] == data_set)], data_set))
  summary = pd.concat(summary)

  num_rows = summary.shape[0]

  summary['sensitivity'] = 0
  summary['specificity'] = 0

  idx = (summary['num_true_positive'] > 0)
  summary.loc[idx, 'sensitivity'] = summary.loc[idx, 'num_corr_positive'] / summary.loc[idx, 'num_true_positive']

  idx = (summary['num_true_negative'] > 0)
  summary.loc[idx, 'specificity'] = summary.loc[idx, 'num_corr_negative'] / summary.loc[idx, 'num_true_negative']

  summary = summary[[ 'data_set', 'event', 'sensitivity', 'specificity' ]]

  return summary

# Get hyper_conf
with hyper_conf_path.open(mode='r') as file:
  hyper_conf = json.load(file)

# Get conf
with conf_path.open(mode='r') as file:
  orig_conf = json.load(file)

params = list(hyper_conf.keys())
param_ranges = []
divisor = steps**(float(1)/len(params))
params_string = []
for param in params:
  min_value = np.min(hyper_conf[param])
  max_value = np.max(hyper_conf[param])
  delta = (max_value-min_value)/divisor
  param_ranges.append(np.arange(min_value, max_value, delta))
  params_string.append("{} = {:2.6f}")

params_string = ", ".join(params_string)
params_string = "\r Checking: " + params_string + " ({}/{}) Avg. time/run: {:2.2f} s"

combinations = np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(params))

results = []

print("Running model \033[1m{}\033[0m...".format(classifier_name))
times = deque([])
for i, combination in enumerate(combinations):
  avg_time = np.sum(times)/len(times) if len(times) > 0 else 0
  start = time.time()

  conf = orig_conf.copy()
  pretty_param_vals = []
  for j,param in enumerate(params):
    conf[param] = combination[j]
    pretty_param_vals.append(param)
    pretty_param_vals.append(combination[j])
  pretty_param_vals.append(i)
  pretty_param_vals.append(len(combinations))
  pretty_param_vals.append(avg_time)

  print(params_string.format(*pretty_param_vals), end="\r")

  summary = run_model(classifier, conf, data)
  for j,param in enumerate(params):
    summary[param] = combination[j]

  results.append(summary)
  times.append(time.time()-start)
  if len(times) > 10:
    times.popleft()

print()
result = pd.concat(results)

output_path.mkdir(exist_ok=True)
csv_path = (output_path / "auc.csv")
result.to_csv(str(output_path / "auc.csv"), header=True, encoding='utf-8', index=None)

if skip_graphs is not True:
  graph_path = (output_path / "auc.pdf")
  print("Printing graphs to \033[1m" + str(graph_path) + "\033[0m")
  cmd = [
    "Rscript",
    "--vanilla",
    str(R_GRAPH_PATH),
    str(csv_path),
    str(graph_path)
  ]
  subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)