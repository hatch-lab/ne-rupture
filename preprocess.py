# coding=utf-8

"""
Gets data into a format readable by classifiers

Usage:
  preprocess.py PROCESSOR INPUT [--output-dir=<string>] [--img-dir=<string>] [--mip-dir=<string>] [--output-name=<string>] [--data-dir=<string>] [--frame-rate=<int>] [--filter-window=<int>] [--gamma=<float>] [--channel=<int>] [--data-set=<string>] [--pixel-size=<int>] [--keep-imgs] [--rolling-ball-size=<int>]

Arguments:
  PROCESSOR The kind of image processor to use (eg, imaris or matlab)
  INPUT Path to the directory containing the raw data (CSV files for Imaris, TIFFs for MATLAB)

Options:
  -h --help Show this screen.
  --output-dir=<string>  [default: input] The subdirectory to save the resulting CSV file
  --output-name=<string>  [default: data.csv] The name of the resulting CSV file
  --img-dir=<string>  [defaults: INPUT/images/(data_set)] The path to TIFF files
  --mip-dir=<string>  [defaults: INPUT/images/(data_set)/mip] The path to MIP files
  --data-dir=<string>  Where to find the raw data. Typically determined by the preprocessor you've selected.
  --filter-window=<int>  [default: 8] The window size used for the median pass filter, in px
  --gamma=<float>  [default: 0.50] The gamma correction to use
  --channel=<int>  [default: 1] The channel to keep (ie, the NLS-3xGFP channel)
  --data-set=<string>  The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<float>  [default: 1] Pixels per micron. If 0, will attempt to detect automatically.
  --rolling-ball-size=<int>  [default: 100] The rolling ball diameter to use for rolling ball subtraction, in um
  --frame-rate=<int>  [default: 180] The seconds that elapse between frames
  --keep-imgs  Whether to store an image of each particle for each frame

Output:
  A CSV file with processed data
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

import re

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

arguments = docopt(__doc__, version=get_version())

schema = Schema({
  'PROCESSOR': And(len, lambda n: (ROOT_PATH / ('preprocessors/' + str(n) + '.py')).is_file(), error='That preprocessor does not exist'),
  'INPUT': And(len, lambda n: os.path.exists(n), error='That folder does not exist'),
  '--output-dir': len,
  '--output-name': len,
  '--img-dir': Or(None, len),
  '--mip-dir': Or(None, len),
  '--data-dir': Or(None, len),
  '--filter-window': And(Use(int), lambda n: n > 0, error='--filter-window must be > 0'),
  '--gamma': And(Use(float), lambda n: n > 0, error='--gamma must be > 0'),
  '--channel': And(Use(int), lambda n: n > 0),
  '--data-set': Or(None, len),
  '--pixel-size': And(Use(float), lambda n: n >= 0, error='--pixel-size must be > 0'),
  '--rolling-ball-size': And(Use(int), lambda n: n > 0),
  '--frame-rate': And(Use(int), lambda n: n > 0),
  Optional('--keep-imgs'): bool
})

try:
  arguments = schema.validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

### Arguments and inputs
processor = import_module("preprocessors." + arguments['PROCESSOR'])

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / (arguments['--output-dir']))
output_name = arguments['--output-name']
data_path = (ROOT_PATH / (arguments['--data-dir'])).resolve() if arguments['--data-dir'] else processor.get_default_data_path(input_path)

data_set = arguments['--data-set'] if arguments['--data-set'] else (input_path).name
frame_rate = arguments['--frame-rate']

tiff_path = (input_path / (arguments['--img-dir'])).resolve() if arguments['--img-dir'] else (input_path / ("images/" + data_set))
mip_path = (input_path / (arguments['--mip-dir'])).resolve() if arguments['--mip-dir'] else (tiff_path / "mip")

filter_window = arguments['--filter-window']
gamma = arguments['--gamma']
channel = arguments['--channel']
pixel_size = arguments['--pixel-size'] if arguments['--pixel-size'] > 0 else None
rolling_ball_size = arguments['--rolling-ball-size']

keep_imgs = bool(arguments['--keep-imgs']) if arguments['--keep-imgs'] else False


### Preprocess our data
params = {
  'data_set': data_set,
  'input_path': input_path,
  'tiff_path': tiff_path,
  'mip_path': mip_path,
  'frame_rate': frame_rate,
  'filter_window': filter_window,
  'gamma': gamma,
  'channel': channel,
  'pixel_size': pixel_size,
  'rolling_ball_size': rolling_ball_size,
  'keep_imgs': keep_imgs
}

data = processor.process_data(data_path, params)

output_path.mkdir(exist_ok=True)
output_file_path = (output_path / (output_name)).resolve()

data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

exit()