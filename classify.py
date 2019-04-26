# coding=utf-8

"""
Classifies particle events using a given classifier

Usage:
  classify.py CLASSIFIER INPUT OUTPUT [--input-name=data.csv] [--output-name=results.csv] [--skip-graphs=0] [--img-dir=0] [--conf=0] [--max-processes=None]

Arguments:
  CLASSIFIER The name of the classifier to test
  INPUT Path to the directory containing the processed output file
  OUTPUT Path to where the classified data CSV file should be saved

Options:
  -h --help Show this screen.
  --version Show version.
  --input-name=<string> [defaults: data.csv] The name of the input CSV file
  --output-name=<string> [defaults: results.csv] The name of the resulting CSV file
  --skip-graphs=<bool> [defaults: False] If True, won't output graphs or videos
  --img-dir=<string> [defaults: INPUT/../images] The directory that contains TIFF images of each frame, for outputting videos.
  --conf=<string> [defaults: None] Override configuration options in conf.json with a JSON string.

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

import numpy as np
import pandas as pd
import re
import subprocess

arguments = docopt(__doc__, version=get_version())

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()

### Arguments and inputs
classifier_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['CLASSIFIER'])
if classifier_name != arguments['CLASSIFIER']:
  print(colorize("yellow", "Classifier input has been sanitized to " + classifier_name))

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (ROOT_PATH / (arguments['OUTPUT'])).resolve()
data_file_path = input_path / (arguments['--input-name']) if arguments['--input-name'] else input_path / "data.csv"
tiff_path = input_path / (arguments['--img-dir']) if arguments['--img-dir'] else (input_path / ("../images/")).resolve()
output_name = arguments['--output-name'] if arguments['--output-name'] else "results.csv"
skip_graphs = True if arguments['--skip-graphs'] else False
conf = json.loads(arguments['--conf']) if arguments['--conf'] else False

### Get our classifier
if not (ROOT_PATH / ("classifiers/" + classifier_name + ".py")).exists():
  print(colorize("red", "No such classifier exists"))

classifier = import_module("classifiers." + classifier_name)

### Read our data
data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

classified_data = classifier.run(data, conf)

output_path.mkdir(exist_ok=True)
output_file_path = (output_path / (output_name)).resolve()

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

