# coding=utf-8

"""
Attempts to find optimal parameters for a given model

Usage:
  find-hyperparameters.py CLASSIFIER [--verbose=0]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --verbose=<Bool> Whether to print the output from the classifier

Output:
  CSV file with parameter, TP, and FN rates for drawing AUC
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt
from common.version import get_version
from common.output import colorize

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
MAX_PROCESSES     = 4

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

classifier = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier))

verbose = True if arguments['--verbose'] else False

hyper_conf_path = (ROOT_PATH / ("classifiers/" + classifier + "/hyperparameters.conf.json")).resolve()
conf_path       = (ROOT_PATH / ("classifiers/" + classifier + "/conf.json")).resolve()
output_path     = (ROOT_PATH / ("validate/output/" + classifier)).resolve()

def run_model(validate_path, classifier, output_path, conf_path, param, value):
  """
  
  """
  # Get conf
  with conf_path.open(mode='r') as file:
    conf = json.load(file)

  # Sample possible values for each parameter in [hyper_conf]
  conf[param] = value

  cmd = [
    "python",
    str(validate_path),
    classifier,
    "--classifier-conf=" + json.dumps(conf)
  ]

  try:
    if verbose:
      subprocess.check_call(cmd)
    else:
      out = subprocess.check_output(cmd)
  except:
    print(colorize("red", "Could not run classifier \033[1m" + classifier + "\033[0m"))
    exit(1)

  results_file_path = output_path / "summary.csv"
  summary = pd.read_csv(str(results_file_path), header=0)

  num_rows = summary.shape[0]

  summary['sensitivity'] = 0
  summary['specificity'] = 0

  idx = (summary['num_true_positive'] > 0)
  summary.loc[idx, 'sensitivity'] = summary.loc[idx, 'num_corr_positive'] / summary.loc[idx, 'num_true_positive']

  idx = (summary['num_true_negative'] > 0)
  summary.loc[idx, 'specificity'] = summary.loc[idx, 'num_corr_negative'] / summary.loc[idx, 'num_true_negative']

  summary = summary[[ 'data_set', 'event', 'sensitivity', 'specificity' ]]
  summary['param'] = param
  summary['value'] = value

  return summary

# Get hyper_conf
with hyper_conf_path.open(mode='r') as file:
  hyper_conf = json.load(file)

params_left = list(hyper_conf.keys())
current_param = params_left.pop(0)

results = []

print("Running model \033[1m{}\033[0m...".format(classifier))
while(current_param is not None):
  max_value = np.max(hyper_conf[current_param])
  min_value = np.min(hyper_conf[current_param])
  current_param_value = float(min_value)
  delta = (max_value - min_value) / 60
  while(current_param_value <= max_value):
    if not verbose:
      progress = int(40*(current_param_value-min_value)/(max_value-min_value))
      bar = "#" * progress + ' ' * (40 - progress)
      print("\r  Checking {} |{}| ({:2.4f})".format(current_param, bar, current_param_value), end="\r")

    results.append(run_model(VALIDATE_PATH, classifier, output_path, conf_path, current_param, current_param_value))
    current_param_value = current_param_value + delta

  bar = "#" * 40
  print("\r  Checking {} |{}| ({:.4f})".format(current_param, bar, current_param_value), end="\r")
  print()

  current_param = params_left.pop(0) if len(params_left) > 0 else None

result = pd.concat(results)
result = pd.DataFrame(result)
result.to_csv(str(output_path / "auc.csv"), header=True, encoding='utf-8', index=None)

# Finish out by restoring original summary/cell-data stats
cmd = [
  "python",
  str(VALIDATE_PATH),
  classifier
]
try:
  if verbose:
    subprocess.check_call(cmd)
  else:
    out = subprocess.check_output(cmd)
except:
  print(colorize("red", "Could not run restore summary.csv and cell-data.csv for \033[1m" + classifier + "\033[0m"))
  exit(1)