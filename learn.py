# coding=utf-8

"""
Attempts to find optimal parameters for a given classifier

Usage:
  find-hyperparameters.py CLASSIFIER INPUT [options]

Arguments:
  CLASSIFIER The name of the classifier to test
  INPUT The gold standard, already classified data to use

Options:
  -h,--help Show this screen.
  -v, --version Show version.
  --steps=<int> [defaults: 60] The number of values to try for each hyper parameter
  --output-dir=<string>  [default: output] The name of the subdirectory in which to store output
  --csv-name=<string>  [default: results.csv] The name of the resulting CSV file
  --img-dir=<string>  [default: images] The subdirectory that contains TIFF images of each frame, for outputting videos.

Output:
  CSV file with TP, and FN rates for drawing AUC
"""
import sys
import os
from pathlib import Path
import shutil
from importlib import import_module

ROOT_PATH = Path(__file__ + "/..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
from lib.summarize import get_cell_stats,get_summary_table

from collections import deque
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
from tqdm import tqdm

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

### Constants
R_GRAPH_PATH = (ROOT_PATH / ("lib/R/learn.R")).resolve()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

classifier_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier_name != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier has been sanitized to " + classifier_name))

if len(classifier_name) <= 0 or not ((ROOT_PATH / ('classifiers/' + str(classifier_name) + '.py'))).is_file():
  print(colorize("red", 'That classifier does not exist.'))

schema = {
  'CLASSIFIER': And(len, lambda n: (ROOT_PATH / ('classifiers/' + str(n) + '.py')).is_file(), error='That classifier does not exist'),
  'INPUT': And(len, lambda n: os.path.exists(n), error='INPUT does not exist'),
  '--csv-name': len,
  '--output-dir': len,
  '--img-dir': len,
  '--steps': And(Use(int), lambda x: x > 0),
  Optional('--skip-graphs'): bool,
  '--help': Or(None, bool),
  '--version': Or(None, bool)
}

try:
  arguments = Schema(schema).validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

steps = int(arguments['--steps']) if arguments['--steps'] else 3600
skip_graphs = bool(arguments['--skip-graphs']) if arguments['--skip-graphs'] else False

input_root = (ROOT_PATH / (arguments['INPUT'])).resolve()
data_file_path = input_root / (arguments['--output-dir'] + '/' + arguments['--csv-name'])
if not data_file_path.exists() or data_file_path.is_file():
  print(colorize("red", "No CSV file can be found"))
  exit(1)

hyper_conf_path = (ROOT_PATH / ("classifiers/" + classifier_name + "/hyperparameters.conf.json")).resolve()
conf_path       = (ROOT_PATH / ("classifiers/" + classifier_name + "/conf.json")).resolve()
tiff_path       = input_root / (arguments['--img-dir'])

now = datetime.now()
output_path     = input_root / ("learn/" + classifier_name + "/" + now.strftime("%Y-%m-%d"))

output_path.mkdir(mode=0o755, exist_ok=True, parents=True)

### Get our classifier
classifier = import_module("classifiers." + classifier_name)

data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

# Set up our gold standard data
data['true_event'] = data['event']
data['true_event_id'] = data['event_id']

data['event'] = 'N'
data['event_id'] = -1

def run_model(classifier, conf, data):
  classified_data = classifier.run(data, tiff_path, conf=conf, fast=True)

  # Get summary data
  results = classified_data.groupby([ 'data_set', 'particle_id' ]).apply(get_cell_stats)
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
  params_string.append("{} = {:2.4f}")

params_string = ", ".join(params_string)
params_string = "Running {}: " + params_string

combinations = np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(params))

results = []

with tqdm(total=len(combinations)) as pbar:
  for i, combination in enumerate(combinations): 
    conf = orig_conf.copy()
    pretty_param_vals = [ classifier_name ]

    for j,param in enumerate(params):
      conf[param] = combination[j]
      pretty_param_vals.append(param)
      pretty_param_vals.append(combination[j])
    pretty_param_vals.append(i)
    pretty_param_vals.append(len(combinations))

    pbar.set_description(params_string.format(*pretty_param_vals))
    summary = run_model(classifier, conf, data)
    for j,param in enumerate(params):
      summary[param] = combination[j]

    results.append(summary)

    pbar.update(1)

result = pd.concat(results)

output_path.mkdir(exist_ok=True)
auc_path = (output_path / "auc.csv")
result.to_csv(str(auc_path), header=True, encoding='utf-8', index=None)
shutil.copy(str(hyper_conf_path), str(output_path / "hyperparameters.conf.json"))

graph_path = (output_path / "auc.pdf")
print("Printing graphs to \033[1m" + str(graph_path) + "\033[0m")
cmd = [
  "Rscript",
  "--vanilla",
  str(R_GRAPH_PATH),
  str(csv_path),
  str(graph_path)
]
subprocess.call(cmd)