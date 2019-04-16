# coding=utf-8

"""
Calculates scores for a given classifier

Usage:
  validate.py CLASSIFIER [--test-data-folder=validate/validation-data/input/] [--classifier-conf=0] [--max-processes=4] [--skip-graphs=0] [--skip-filtered=0]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --test-data-folder=<string> [defaults: validate/validation-data/input] The directory with the CSV file containing particle data with true events
  --classifier-conf=<string> [defaults: None] Will be passed along to the classifier.
  --max-processes=<int> [defaults: 4] The number of threads we will allow the classifier to use.
  --skip-graphs=<bool> [defaults: False] If True, won't output graphs or videos
  --skip-filtered=<bool> [defaults: False] If True, won't include filtered data in summary stats

Output:
  Prints alignment scores, their mean, and their SD.
  Generates graphs of each nucleus's predicted and actual events.
  Generates annotated videos of each nucleus with either a predicted or a true event.
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt
from common.version import get_version

import numpy as np
import pandas as pd
import csv
import subprocess
import math
import re
from tabulate import tabulate
from multiprocessing import Pool, cpu_count

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

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()

STATS_EVENT_MAP = {
  "R": "Rupture",
  "M": "Mitosis",
  "X": "Apoptosis"
}

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

classifier = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier))

data_file_path = Path(arguments['--test-data-folder']).resolve() if arguments['--test-data-folder'] else Path("validate/validation-data/input/").resolve()
if not data_file_path.exists():
  print(colorize("red", "Data folder input cannot be found: \033[1m" + str(data_file_path) + "\033[0m"))
  exit(1)

classifier_conf = arguments['--classifier-conf'] if arguments['--classifier-conf'] else "0"

max_processes = int(arguments['--max-processes']) if arguments['--max-processes'] else 4

skip_graphs = bool(arguments['--skip-graphs']) if arguments['--skip-graphs'] else False
skip_filtered = bool(arguments['--skip-filtered']) if arguments['--skip-filtered'] else False

classifier_path = (ROOT_PATH / ("classifiers/" + classifier + ".py")).resolve()
output_path     = (ROOT_PATH / ("validate/output/" + classifier)).resolve()
output_path.mkdir(exist_ok=True)

### Run prediction
print("Running classifier \033[1m" + classifier + "\033[0m...")
results_path = (output_path).resolve()
cmd = [
  "python",
  str(classifier_path),
  str(data_file_path),
  str(results_path),
  "--conf=" + classifier_conf,
  "--max-processes=" + str(max_processes),
  "--skip-graphs=" + str(skip_graphs)
]
try:
  subprocess.check_call(cmd)
except:
  print(colorize("red", "Could not run classifier \033[1m" + classifier + "\033[0m"))
  exit(1)

### Read our predicted data
results_file_path = output_path / "results.csv"
data = pd.read_csv(str(results_file_path), header=0, dtype={ 'particle_id': str })

def get_cell_stats(p_data, skip_filtered):
  data_set = p_data['data_set'].iloc[0]
  particle_id = p_data['particle_id'].iloc[0]

  result = pd.DataFrame({
    'data_set': [ data_set ],
    'particle_id': [ particle_id ]
  })
  
  if skip_filtered and 'filtered' in p_data.columns:
    p_data = p_data.loc[(p_data['filtered'] == 0)]

  for event,name in STATS_EVENT_MAP.items():
    result['pred' + event] = True if event in p_data['event'].unique() else False
    result['true' + event] = True if event in p_data['true_event'].unique() else False

  return result

def get_summary_table(results, data_set):
  names = []
  nums_corr_positive = []
  nums_pred_positive = []
  nums_true_positive = []
  nums_corr_negative = []
  nums_pred_negative = []
  nums_true_negative = []
  
  for event,name in STATS_EVENT_MAP.items():
    names.append(name)
    nums_corr_positive.append(results[((results['pred' + event] == True) & (results['true' + event] == True))].shape[0])
    nums_pred_positive.append(results[((results['pred' + event] == True))].shape[0])
    nums_true_positive.append(results[((results['true' + event] == True))].shape[0])
    nums_corr_negative.append(results[((results['pred' + event] == False) & (results['true' + event] == False))].shape[0])
    nums_pred_negative.append(results[((results['pred' + event] == False))].shape[0])
    nums_true_negative.append(results[((results['true' + event] == False))].shape[0])
  
  summary = pd.DataFrame({
    'data_set': [data_set] * len(names),
    'event': names,
    'num_corr_positive': nums_corr_positive,
    'num_pred_positive': nums_pred_positive,
    'num_true_positive': nums_true_positive,
    'num_corr_negative': nums_corr_negative,
    'num_pred_negative': nums_pred_negative,
    'num_true_negative': nums_true_negative
  })

  return summary

def prettify_summary_table(summary):
  data_sets = []
  events = []
  true_positives = [] # Sensitivity
  false_positives = [] # Fall-out
  true_negatives = [] # Specificity
  false_negatives = [] # Miss
  ppv = [] # Precision
  npv = []
  fdr = []

  for index,row in summary.iterrows():
    true_positive_rate = row['num_corr_positive']/row['num_true_positive'] if row['num_true_positive'] > 0 else 0
    false_positive_rate = 1-true_positive_rate if true_positive_rate > 0 else 0
    true_negative_rate = row['num_corr_negative']/row['num_true_negative'] if row['num_pred_negative'] > 0 else 0
    false_negative_rate = 1-true_negative_rate if true_negative_rate > 0 else 0
    ppv_rate = row['num_corr_positive']/row['num_pred_positive'] if row['num_pred_positive'] > 0 else 0
    npv_rate = row['num_corr_negative']/row['num_pred_negative'] if row['num_pred_negative'] > 0 else 0
    fdr_rate = 1-ppv_rate if ppv_rate > 0 else 0

    events.append(row['event'])
    data_sets.append(row['data_set'])
    true_positives.append("{:.2%} ({}/{})".format(true_positive_rate, row['num_corr_positive'], row['num_true_positive']))
    false_positives.append("{:.2%} ({}/{})".format(false_positive_rate, (row['num_true_positive']-row['num_corr_positive']), row['num_true_positive']))
    true_negatives.append("{:.2%} ({}/{})".format(true_negative_rate, row['num_corr_negative'], row['num_true_negative']))
    false_negatives.append("{:.2%} ({}/{})".format(false_negative_rate, (row['num_true_negative']-row['num_corr_negative']), row['num_true_negative']))
    ppv.append("{:.2%}, ({}/{})".format(ppv_rate, row['num_corr_positive'], row['num_pred_positive']))
    npv.append("{:.2%}, ({}/{})".format(npv_rate, row['num_corr_negative'], row['num_pred_negative']))
    fdr.append("{:.2%}, ({}/{})".format(fdr_rate, (row['num_pred_positive']-row['num_corr_positive']), row['num_pred_positive']))

  pretty = pd.DataFrame({
    'data_set': data_sets,
    'event': events,
    'true_positive': true_positives,
    'false_positives': false_positives,
    'true_negatives': true_negatives,
    'false_negatives': false_negatives,
    'ppv': ppv,
    'npv': npv,
    'fdr': fdr
  })

  return pretty

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
  with Pool(cpu_count()) as p:
    groups = []
    for name, group in grouped:
      t = tuple([ group ]) + tuple(args)
      groups.append(t)
    chunk = p.starmap(fn, groups)

  return chunk

if __name__ == '__main__':
  print("Scoring classifier \033[1m" + classifier + "\033[0m...")
  results = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), get_cell_stats, skip_filtered)
  results = pd.concat(results)

  headers = [
    "",
    "Event", 
    "True positives", 
    "False positives", 
    "True negatives", 
    "False negatives", 
    "Pos. Predictive Value", 
    "Neg. Predictive Value", 
    "FDR"
  ]

  summary_table = []
  print("All:")
  print("{} cells".format(results.shape[0]))
  summary_table.append(get_summary_table(results, "All"))
  print(tabulate(prettify_summary_table(summary_table[-1]), headers=headers))
  print()
  print("By dataset:")
  for data_set in results['data_set'].unique():
    summary_table.append(get_summary_table(results[(results['data_set'] == data_set)], data_set))
    print("  \033[1m" + data_set + "\033[0m")
    print("  {} cells".format(results[(results['data_set'] == data_set)].shape[0]))
    print(tabulate(prettify_summary_table(summary_table[-1]), headers=headers))
    print()

  summary_table = pd.concat(summary_table)

  summary_table.to_csv(str(results_path / "summary.csv"), header=True, encoding='utf-8', index=None)
  results.to_csv(str(results_path / "cell-results.csv"), header=True, encoding='utf-8', index=None)

exit()

