# coding=utf-8

"""
Calculates scores for a given classifier

Usage:
  validate.py CLASSIFIER [--img-dir=dir] [--test-data-file=validate/validation-data/validation-data.csv] [--classifier-conf=0] [--simple-output=0] [--max-processes=4]

Arguments:
  CLASSIFIER The name of the classifier to test

Options:
  --img-dir=<string|None> [defaults: None] The directory that contains TIFF images of each frame, for outputting videos.
  --test-data-file=<string> [defaults: validation-data.csv] The CSV file containing particle data with true events
  --classifier-conf=<string> [defaults: None] Will be passed along to the classifier.
  --simple-output=<int> [defaults: False] If simple, machine-readable output is desired.
  --max-processes=<int> [defaults: 4] The number of threads we will allow the classifier to use.

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

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
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

def _print(silence, string):
  if not silence:
    print(string)

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

simple_output = True if arguments['--simple-output'] else False

classifier = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier != arguments['CLASSIFIER']:
  _print(simple_output, colorize("yellow", "Classifier input has been sanitized to " + classifier))

data_file_path = Path(arguments['--test-data-file']).resolve() if arguments['--test-data-file'] else Path("validate/validation-data/validation-data.csv").resolve()
if not data_file_path.exists():
  _print(simple_output, colorize("red", "Data file input cannot be found: \033[1m" + str(data_file_path) + "\033[0m"))
  exit(1)

tiff_path = Path(arguments['--img-dir']).resolve() if arguments['--img-dir'] else False
if tiff_path and not tiff_path.is_dir():
  _print(simple_output, colorize("red", "The supplied img-dir does not exist: \033[1m" + str(tiff_path) + "\033[0m"))
  exit(1)
elif tiff_path:
  tiff_path = str(tiff_path)

classifier_conf = arguments['--classifier-conf'] if arguments['--classifier-conf'] else "0"

max_processes = int(arguments['--max-processes']) if arguments['--max-processes'] else 4

classifier_path = (ROOT_PATH / ("classifiers/" + classifier + ".py")).resolve()
output_path     = (ROOT_PATH / ("validate/output/" + classifier)).resolve()
output_path.mkdir(exist_ok=True)

### Run prediction
_print(simple_output, "Running classifier \033[1m" + classifier + "\033[0m...")
results_path = (output_path).resolve()
cmd = [
  "python",
  str(classifier_path),
  str(data_file_path),
  str(results_path),
  "--conf=" + classifier_conf,
  "--max-processes=" + str(max_processes)
]
try:
  if simple_output:
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  else:
    subprocess.check_call(cmd)
except:
  print(colorize("red", "Could not run classifier \033[1m" + classifier + "\033[0m"))
  exit(1)

### Read our predicted data
results_file_path = output_path / "results.csv"
data = pd.read_csv(str(results_file_path), header=0, dtype={ 'particle_id': str })
data = data[[ 'data_set', 'particle_id', 'time', 'event', 'true_event' ]]


### Score our data
# We will use sequence alignment algorithms to align a particles events with true_events
# and score the alignment

def score_sequence(p_data): 
  """
  Score using sequence alignment algorithm

  Arguments:
    p_data Pandas DataFrame The data for a single particle

  Returns:
    Alignment score
  """
  _print(simple_output, "  Scoring \033[1m" + p_data['data_set'].unique()[0] + ':' + p_data['particle_id'].unique()[0] + "\033[0m")
  p_data = p_data.sort_values(by=[ 'time' ])
  true = "".join(list(p_data['true_event']))
  pred = "".join(list(p_data['event']))

  score_key = {
    ('N', 'N'): 0.0001,
    ('N', 'R'): -5,
    ('N', 'E'): -5,
    ('N', 'M'): -2,
    ('N', 'X'): -2,

    ('R', 'R'): 2,
    ('R', 'N'): -1,
    ('R', 'E'): 1.5,
    ('R', 'M'): -2,
    ('R', 'X'): -2,

    ('E', 'E'): 2,
    ('E', 'N'): -1,
    ('E', 'R'): 1.5,
    ('E', 'M'): -2,
    ('E', 'X'): -2,

    ('M', 'M'): 2,
    ('M', 'N'): 0,
    ('M', 'R'): -2,
    ('M', 'E'): -2,
    ('M', 'X'): -2,

    ('X', 'X'): 2,
    ('X', 'N'): 0,
    ('X', 'R'): -2,
    ('X', 'E'): -2,
    ('X', 'M'): -2
  }
  alignments = pairwise2.align.globalds(true, pred, score_key, -1, -0.5)
  # perfect = pairwise2.align.globalds(true, true, score_key, -1, -0.5)
  if len(alignments) >= 1:
    return alignments[0][2]

  return -89

# def score_event(true, pred):
#   """
#   Score the true vs predicted event

#   Gives very little weight to correctly predicting a non-event 
#   frame. Also gives little weight if the true frame carries 
#   death events, since we can't yet predict those.

#   Arguments:
#     true The true frame event
#     pred The predicted frame event

#   Returns:
#     The score
#   """
#   if pred == 'N' and (true == 'X' or true == 'M'):
#     return 0.01

#   if true == pred:
#     return 0.01 if true == 'N' else 5

#   if true != 'N':
#     # Penalize false positives
#     return -8

#   return -1

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
      # We're going to skip sequences where true and pred
      # values are entirely N
      if (group['event'][( group['event'] != 'N' )].count() <= 0) and (group['event'][( group['true_event'] != 'N' )].count() <= 0):
        _print(simple_output, "  Skipping \033[1m" + group['data_set'].unique()[0] + ':' + group['particle_id'].unique()[0] + "\033[0m")
        continue
      t = tuple([ group ]) + tuple(args)
      groups.append(t)
    chunk = p.starmap(fn, groups)

  return chunk

if __name__ == '__main__':
  _print(simple_output, "Scoring classifier \033[1m" + classifier + "\033[0m...")
  test_results = apply_parallel(data.groupby([ 'data_set', 'particle_id' ]), score_sequence)

if tiff_path:
  cmd = [
    "python",
    str(QA_PATH),
    classifier,
    str(results_file_path),
    str(output_path),
    "--img-dir=" + tiff_path
  ]
  if simple_output:
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  else:
    subprocess.call(cmd)

_print(simple_output, "")
if simple_output:
  print(str(np.median(test_results)) + " " + str(np.std(test_results)))
else:
  print("\033[1mMedian:\033[0m {:.3f}".format(np.median(test_results)))
  print("\033[1mStd. dev.:\033[0m {:.3f}".format(np.std(test_results)))
_print(simple_output, "")

exit()

