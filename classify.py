# coding=utf-8

"""
Classifies particle events using a given classifier

Usage:
  classify.py CLASSIFIER INPUT [--input-name=<string>] [--output-name=<string>] [--output-dir=<string>] [--input-dir=<string>] [--skip-graphs] [--img-dir=<string>] [--conf=<string>]

Arguments:
  CLASSIFIER The name of the classifier to test
  INPUT Path to the directory containing particle data

Options:
  -h --help Show this screen.
  --version Show version.
  --input-name=<string>  [default: data.csv] The name of the input CSV file
  --output-name=<string>  [default: results.csv] The name of the resulting CSV file
  --output-dir=<string>  [default: output] The name of the subdirectory in which to store output
  --input-dir=<string>  [default: input] The name of the subdirectory in which to find the inpute CSV file
  --skip-graphs  Whether to skip producing graphs or videos
  --img-dir=<string>  [default: images] The subdirectory that contains TIFF images of each frame, for outputting videos.
  --conf=<string>  Override configuration options in conf.json with a JSON string.

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

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

arguments = docopt(__doc__, version=get_version())

schema = Schema({
  'CLASSIFIER': And(len, lambda n: (ROOT_PATH / ('classifiers/' + str(n) + '.py')).is_file(), error='That classifier does not exist'),
  'INPUT': And(len, lambda n: os.path.exists(n), error='INPUT does not exist'),
  '--input-name': len,
  '--output-name': len,
  '--output-dir': len,
  '--input-dir': len,
  Optional('--skip-graphs'): bool,
  '--img-dir': len,
  '--conf': Or(None, len)
})

try:
  arguments = schema.validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

### Constant for getting our base input dir
QA_PATH  = (ROOT_PATH / ("validate/qa.py")).resolve()

### Arguments and inputs
classifier_name = arguments['CLASSIFIER']

input_root = (ROOT_PATH / (arguments['INPUT'])).resolve()
input_path = input_root / (arguments['--input-dir'])
output_path = input_root / (arguments['--output-dir'])
tiff_path = input_root / (arguments['--img-dir'])

data_file_path = input_path / (arguments['--input-name'])
output_name = arguments['--output-name']
skip_graphs = True if arguments['--skip-graphs'] else False
conf = json.loads(arguments['--conf']) if arguments['--conf'] else False

### Get our classifier
classifier = import_module("classifiers." + classifier_name)

### Read our data
data = pd.read_csv(str(data_file_path), header=0, dtype={ 'particle_id': str })

if classifier.NEED_TIFFS:
  classified_data = classifier.run(data, tiff_path, conf=conf)
else:
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

