# coding=utf-8

"""
Calculates scores for a given classifier

Usage:
  validate.py CLASSIFIER [--input-path=validate/validation-data/input/] [--input-name=data.csv] [--img-dir=0] [--classifier-conf=0] [--skip-graphs=0] [--skip-filtered=0]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --input-path=<string> [defaults: validate/validation-data/input] The directory with the CSV file containing particle data with true events
  --input-name=<string> [defaults: data.csv] The name of the input CSV file
  --img-dir=<string> [defaults: [input_path]/../images] The directory that contains TIFF images of each frame, for outputting videos.
  --classifier-conf=<string> [defaults: None] Will be passed along to the classifier.
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
from importlib import import_module

ROOT_PATH = Path(__file__ + "/..").resolve()
sys.path.append(str(ROOT_PATH))


from common.docopt import docopt
from common.version import get_version
from common.output import colorize

from validate.lib import get_cell_stats,get_summary_table

import numpy as np
import pandas as pd
import subprocess
import math
import re
from tabulate import tabulate
from tqdm import tqdm

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

classifier_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier_name != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier_name))

input_path = Path(arguments['--input-path']).resolve() if arguments['--input-path'] else Path("validate/validation-data/input").resolve()
if not input_path.exists():
  print(colorize("red", "Data folder input cannot be found: \033[1m" + str(input_path) + "\033[0m"))
  exit(1)

data_file_path = input_path / (arguments['--input-name']) if arguments['--input-name'] else input_path / "data.csv"
tiff_path = input_path / (arguments['--img-dir']) if arguments['--img-dir'] else (input_path / ("../images/")).resolve()

classifier_conf = arguments['--classifier-conf'] if arguments['--classifier-conf'] else False

skip_graphs = bool(arguments['--skip-graphs']) if arguments['--skip-graphs'] else False
skip_filtered = bool(arguments['--skip-filtered']) if arguments['--skip-filtered'] else False

# Get paths
classifier_path = (ROOT_PATH / ("classifiers/" + classifier_name + ".py")).resolve()
output_path     = (ROOT_PATH / ("validate/output/" + classifier_name)).resolve()

### Get our classifier
if not (ROOT_PATH / ("classifiers/" + classifier_name + ".py")).exists():
  print(colorize("red", "No such classifier exists"))

classifier = import_module("classifiers." + classifier_name)

### Run prediction
print("Running classifier \033[1m" + classifier_name + "\033[0m...")

data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

finished, classified_data = classifier.run(data, tiff_path, conf=classifier_conf)

def prettify_summary_table(summary):
  data_sets = []
  events = []
  accuracies = []
  true_positives = [] # Sensitivity
  false_positives = [] # Fall-out
  true_negatives = [] # Specificity
  false_negatives = [] # Miss
  ppv = [] # Precision
  npv = []
  fdr = []

  for index,row in summary.iterrows():
    accuracy = (row['num_corr_positive'] + row['num_corr_negative'])/(row['num_true_positive'] + row['num_true_negative'])
    true_positive_rate = row['num_corr_positive']/row['num_true_positive'] if row['num_true_positive'] > 0 else 0
    true_negative_rate = row['num_corr_negative']/row['num_true_negative'] if row['num_pred_negative'] > 0 else 0
    false_positive_rate = 1-true_negative_rate
    false_negative_rate = 1-true_positive_rate
    ppv_rate = row['num_corr_positive']/row['num_pred_positive'] if row['num_pred_positive'] > 0 else 0
    npv_rate = row['num_corr_negative']/row['num_pred_negative'] if row['num_pred_negative'] > 0 else 0
    fdr_rate = 1-ppv_rate

    events.append(row['event'])
    data_sets.append(row['data_set'])
    accuracies.append("{:.2%}".format(accuracy))
    true_positives.append("{:.2%} ({}/{})".format(true_positive_rate, row['num_corr_positive'], row['num_true_positive']))
    false_positives.append("{:.2%} ({}/{})".format(false_positive_rate, (row['num_pred_positive']-row['num_corr_positive']), row['num_true_negative']))
    true_negatives.append("{:.2%} ({}/{})".format(true_negative_rate, row['num_corr_negative'], row['num_true_negative']))
    false_negatives.append("{:.2%} ({}/{})".format(false_negative_rate, (row['num_pred_negative']-row['num_corr_negative']), row['num_true_positive']))
    ppv.append("{:.2%} ({}/{})".format(ppv_rate, row['num_corr_positive'], row['num_pred_positive']))
    npv.append("{:.2%} ({}/{})".format(npv_rate, row['num_corr_negative'], row['num_pred_negative']))
    fdr.append("{:.2%} ({}/{})".format(fdr_rate, (row['num_pred_positive']-row['num_corr_positive']), row['num_pred_positive']))

  pretty = pd.DataFrame({
    'data_set': data_sets,
    'event': events,
    'accuracy': accuracies,
    'true_positives': true_positives,
    'false_positives': false_positives,
    'true_negatives': true_negatives,
    'false_negatives': false_negatives,
    'ppv': ppv,
    'npv': npv,
    'fdr': fdr
  })

  return pretty

# def apply_parallel(grouped, fn, *args):
#   """
#   Function for parallelizing particle classification

#   Will take each DataFrame produced by grouping by particle_id
#   and pass that data to the provided function, along with the 
#   supplied arguments.

#   Arguments:
#     grouped List of grouped particle data
#     fn function The function called with a group as a parameter
#     args Arguments to pass through to fn

#   Returns:
#     Pandas DataFrame The re-assembled data.
#   """
#   with Pool(cpu_count()) as p:
#     groups = []
#     for name, group in grouped:
#       t = tuple([ group ]) + tuple(args)
#       groups.append(t)
#     chunk = p.starmap(fn, groups)

#   return chunk

if __name__ == '__main__':
  print("Scoring classifier \033[1m" + classifier_name + "\033[0m...")
  tqdm.pandas(desc="Generating cell stats", ncols=90, unit="particle")
  results = classified_data.groupby([ 'data_set', 'particle_id' ]).progress_apply(get_cell_stats, skip_filtered=skip_filtered)

  headers = [
    "",
    "Event", 
    "Accuracy",
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

  output_path.mkdir(exist_ok=True)
  summary_table.to_csv(str(output_path / "summary.csv"), header=True, encoding='utf-8', index=None)
  results.to_csv(str(output_path / "cell-results.csv"), header=True, encoding='utf-8', index=None)

  output_file_path = (output_path / "results.csv").resolve()
  classified_data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

  if not skip_graphs:
    cmd = [
      "python",
      str(QA_PATH),
      classifier_name,
      output_file_path,
      str(output_path),
      "--img-dir=" + str(tiff_path)
    ]
    subprocess.call(cmd)

exit()

