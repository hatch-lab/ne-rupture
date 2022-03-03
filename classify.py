# coding=utf-8

"""
Classifies particle events using a given classifier

Usage:
  classify.py [options] CLASSIFIER INPUT [<args>...]

Arguments:
  CLASSIFIER The name of the classifier to test
  INPUT The input path

Options:
  -h --help Show this screen.
  --version Show version.
  --input-name=<string>  [default: data.csv] The name of the input CSV file
  --output-name=<string>  [default: results.csv] The name of the resulting CSV file
  --cell-summary-name=<string>  [default: cell-summary.csv] The name of the CSV file for data summarized by cells
  --event-summary-name=<string>  [default: event-summary.csv] The name of the CSV file for data summarized by events
  --output-dir=<string>  [default: output] The name of the subdirectory in which to store output
  --input-dir=<string>  [default: input] The name of the subdirectory in which to find the inpute CSV file
  --skip-videos  Whether to skip producing graphs or videos
  --img-dir=<string>  [default: images] The subdirectory that contains TIFF images of each frame, for outputting videos.
  --conf=<string>  Override configuration options in conf.json with a JSON string.
  --draw-tracks  [Default: False] Whether to overlay tracks on the annotated movies

Output:
  Generates graphs of each nucleus's predicted and actual events.
  Generates annotated videos of each nucleus with either a predicted or a true event.
"""
import sys
import os
from pathlib import Path, PurePath
from importlib import import_module
import builtins

ROOT_PATH = Path(__file__ + "/..").resolve()
builtins.ROOT_PATH = ROOT_PATH

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
import lib.video as hatchvid
import json
import copy
import re

import numpy as np
import pandas as pd
import subprocess

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

arguments = docopt(__doc__, version=get_version(), options_first=True)

classifier_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier_name != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier has been sanitized to " + classifier_name))

if len(classifier_name) > 0 and os.path.sep not in classifier_name and ((ROOT_PATH / ('classifiers/' + str(classifier_name) + '.py'))).is_file():
  classifier = import_module("classifiers." + classifier_name)
else:
  raise Exception('That classifier does not exist.')

classifier = import_module("classifiers." + classifier_name)

classifier_arguments = docopt(classifier.__doc__, argv=[arguments['CLASSIFIER']] + [arguments['INPUT']] + arguments['<args>'])
classifier_schema = classifier.get_schema()

arguments.update(classifier_arguments)

schema = {
  'CLASSIFIER': And(len, lambda n: (ROOT_PATH / ('classifiers/' + str(n) + '.py')).is_file(), error='That classifier does not exist'),
  'INPUT': And(len, lambda n: os.path.exists(n), error='INPUT does not exist'),
  '--input-name': len,
  '--output-name': len,
  '--event-summary-name': len,
  '--cell-summary-name': len,
  '--output-dir': len,
  '--input-dir': len,
  Optional('--skip-videos'): bool,
  Optional('--draw-tracks'): bool,
  '--img-dir': len,
  '--conf': Or(None, len),
  '--help': Or(None, bool),
  '--version': Or(None, bool),
  Optional('<args>'): lambda n: True,
  Optional('--'): lambda n: True
}
schema.update(classifier_schema)

try:
  arguments = Schema(schema).validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

### Arguments and inputs
classifier_name = arguments['CLASSIFIER']

input_root = (ROOT_PATH / (arguments['INPUT'])).resolve()
input_path = input_root / (arguments['--input-dir'])
output_path = input_root / (arguments['--output-dir'])
tiff_path = input_root / (arguments['--img-dir'])

data_file_path = input_path / (arguments['--input-name'])
output_name = arguments['--output-name']
event_summary_name = arguments['--event-summary-name']
cell_summary_name = arguments['--cell-summary-name']
skip_graphs = True if arguments['--skip-videos'] else False
draw_tracks = True if arguments['--draw-tracks'] else False
conf = json.loads(arguments['--conf']) if arguments['--conf'] else False

start_over = True if '--start-over' in arguments and arguments['--start-over'] else False

### Get our classifier
classifier = import_module("classifiers." + classifier_name)

tmp_path = ROOT_PATH / ("tmp/classifiers/" + classifier_name +  "/" + (re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', input_root.name)))
tmp_path.mkdir(exist_ok=True, mode=0o755, parents=True)
intermediate_path = tmp_path / arguments['--input-name']

if classifier.SAVES_INTERMEDIATES and not start_over:
  data_file_path = intermediate_path if intermediate_path.exists() else data_file_path

### Read our data
data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })
if classifier.SAVES_INTERMEDIATES and 'current_idx' not in data.columns:
  data.loc[:, 'current_idx'] = 0

# Merge in any run-time conf options
if not conf:
  with classifier.CONF_PATH.open(mode='r') as file:
    conf = json.load(file)

if 'tmp_path' not in conf:
  conf['tmp_path'] = str(intermediate_path)

conf.update(arguments)
done, data = classifier.run(data, tiff_path, conf=conf)

output_path.mkdir(exist_ok=True)
output_file_path = (output_path / (output_name)).resolve()

if classifier.SAVES_INTERMEDIATES and not done:
  output_file_path = intermediate_path

data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

if done and intermediate_path.exists():
  # Clear the intermediate file
  intermediate_path.unlink()

if not done:
  print(colorize("magenta", "The classifier did not finish or there was an error."))
  exit(0)

event_file_path = (output_path / (event_summary_name)).resolve()
cell_file_path = (output_path / (cell_summary_name)).resolve()

# Generate event/cell summaries
event_summary = classifier.get_event_summary(data, conf)
if isinstance(event_summary, pd.DataFrame):
  event_summary.to_csv(str(event_file_path), header=True, encoding='utf-8', index=None)

cell_summary = classifier.get_cell_summary(data, conf)
if isinstance(cell_summary, pd.DataFrame):
  cell_summary.to_csv(str(cell_file_path), header=True, encoding='utf-8', index=None)

json_path = output_path / "classify.conf.json"
print("Saving configration options to " + str(json_path))
with open(str(json_path), 'w') as fp:
  json_arguments = copy.deepcopy(arguments)
  for key,arg in arguments.items():
    if isinstance(arg, PurePath):
      json_arguments[key] = str(arg)
  fp.write(json.dumps(json_arguments))

print("Finished!")

if not skip_graphs:
  video_path = output_path / "videos"
  video_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  data.loc[:,'frame_path'] = data.apply(lambda x:
    str( (tiff_path / (x.data_set + '/' + str(x.frame).zfill(4) + '.tif')).resolve() ), axis=1
  )

  hatchvid.make_videos(tiff_path, output_file_path, video_path, draw_tracks=draw_tracks)

print("Finished!")