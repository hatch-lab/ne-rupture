# coding=utf-8

"""
Generates videos/graphs

Usage:
  qa.py CLASSIFIER RESULTS OUTPUT [--img-dir=dir]

Arguments:
  CLASSIFIER The classifier used to make predictions
  RESULTS The CSV file containing predictions
  OUTPUT Where graphs/videos should be output

Options:
  --img-dir=(string | 0) [defaults: 0] The directory that contains TIFF images of each frame, for outputting videos.

Output:
  Generates graphs of each nucleus's predicted and actual events.
  Generates annotated videos of each nucleus with either a predicted or a true event.
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt

import numpy as np
import pandas as pd
import csv
import subprocess
import re

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
MAKE_SINGLE_VIDEO_PATH = (ROOT_PATH / ("validate/render-particle-video.py")).resolve()
MAKE_VIDEO_PATH        = (ROOT_PATH / ("validate/render-full-video.py")).resolve()
MAX_PROCESSES          = 4

### Arguments and inputs
arguments = docopt(__doc__, version='NE-classifier 0.1')

classifier = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier))

r_graph_gen_path = (ROOT_PATH / ("classifiers/" + classifier + "/make-graphs.R")).resolve()
if r_graph_gen_path and not r_graph_gen_path.is_file():
  print(colorize("red", "The supplied classifier cannot make graphs: \033[1m" + str(classifier) + "\033[0m"))
  exit(1)

conf_path = (ROOT_PATH / ("classifiers/" + classifier + "/conf.json")).resolve()

tiff_path = Path(arguments['--img-dir']).resolve() if arguments['--img-dir'] else False
if tiff_path and not tiff_path.is_dir():
  print(colorize("red", "The supplied img-dir does not exist: \033[1m" + str(tiff_path) + "\033[0m"))
  exit(1)

results_path = Path(arguments['RESULTS']).resolve()
output_path  = Path(arguments['OUTPUT']).resolve()

output_path.mkdir(exist_ok=True)

graph_path = str((output_path / "graphs.pdf").resolve())
movie_graphs_path = (output_path / "movie_plots/").resolve()
movie_graphs_path.mkdir(exist_ok=True)

data = pd.read_csv(results_path, header=0, dtype={ 'particle_id': str })
if 'true_event' not in data.columns:
  data['true_event'] = 'N'
data = data[[ 'data_set', 'particle_id', 'time', 'event', 'true_event' ]]

print("Printing graphs to \033[1m" + graph_path + "\033[0m")
cmd = [
  "Rscript",
  "--vanilla",
  str(r_graph_gen_path),
  results_path,
  graph_path,
  "300",
  str(movie_graphs_path),
  str(conf_path)
]
subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# subprocess.call(cmd)

if (ROOT_PATH / ("Rplots.pdf")).exists():
  (ROOT_PATH / ("Rplots.pdf")).unlink() # R keeps spitting this out. Not sure why yet.

if str(tiff_path):
  processes = set()

  video_path = (output_path / "videos/").resolve()
  video_path.mkdir(exist_ok=True)

  print("Generating annotated videos to \033[1m" + str(video_path) + "\033[0m")
  data_sets = np.unique(data['data_set'])
  for data_set in data_sets:
    subset = data[( data['data_set'] == data_set )]

    cmd = [
      "python",
      str(MAKE_VIDEO_PATH),
      str(tiff_path),
      results_path,
      data_set,
      str(video_path)
    ]
    subprocess.call(cmd)

    # We will use multi-threading to generate videos faster
    particle_ids = np.unique(subset['particle_id'])
    for particle_id in particle_ids:
      cmd = [
        "python",
        str(MAKE_SINGLE_VIDEO_PATH),
        str(tiff_path),
        results_path,
        data_set,
        particle_id,
        str(video_path),
        "--graph-file=" + str((movie_graphs_path / (data_set + "_" + particle_id + ".tiff")).resolve())
      ]
      processes.add(subprocess.Popen(cmd, stderr=subprocess.STDOUT))

      if len(processes) >= MAX_PROCESSES:
        os.wait()
        processes.difference_update(
          [p for p in processes if p.poll() is not None]
        )

  # Check if all child processes have been closed
  for p in processes:
    if p.poll() is None:
      p.wait()

# Remove the movie plots
for f in movie_graphs_path.iterdir():
  f.unlink()
movie_graphs_path.rmdir()