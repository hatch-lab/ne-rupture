# coding=utf-8

"""
Gets data into a format readable by classifiers

Usage:
  preprocess.py PROCESSOR INPUT [--output-dir=0] [--img-dir=0] [--output-name=data.csv] [--data-dir=0] [--frame-rate=180] [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SD] [--data-set=0] [--pixel-size=0] [--rolling-ball-size=30] [--img-dir=0]

Arguments:
  PROCESSOR The kind of image processor to use (eg, imaris or matlab)
  INPUT Path to the directory containing the raw data (CSV files for Imaris, TIFFs for MATLAB)
  OUTPUT Path to where the classified data CSV file should be saved

Options:
  -h --help Show this screen.
  --output-dir=<string> [defaults: INPUT/output] The directory to save the resulting CSV file
  --output-name=<string> [defaults: data.csv] The name of the resulting CSV file
  --img-dir=<string> [defaults: INPUT/images/data_set] The path to TIFF files
  --data-dir=<string> [defaults: INPUT/images/raw for MATLAB; INPUT/input for Imaris]
  --filter-window=<float> [defaults: 5.0] The window size used for the median pass filter, in px
  --gamma=<float> [defaults: 0.50] The gamma correction to use
  --channel=<int> [defaults: 1] The channel to keep (ie, the NLS-3xGFP channel)
  --objective=<int> [defaults: 20] The microscope objective (eg, 20 for 20x)
  --microscope=<string> [defaults: SD] "SPE" or "SD"
  --data-set=<string|falsey> [defaults: None] The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<int|0> [defaults: 0] Specifying microscope and objective will automatically determine pixel size. If supplied here, that value will be used instead.
  --rolling-ball-size=<int> [defaults: 30] The rolling ball diameter to use for rolling ball subtraction, in um
  --frame-rate=<int> [defaults: 180] The seconds that elapse between frames

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

arguments = docopt(__doc__, version=get_version())
### Arguments and inputs
processor_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['PROCESSOR'])
if processor_name != arguments['PROCESSOR']:
  print(colorize("yellow", "Classifier input has been sanitized to " + processor_name))

if not (ROOT_PATH / ("preprocessors/" + processor_name + ".py")).exists():
  print(colorize("red", "No such processor exists"))
  exit(1)

processor = import_module("preprocessors." + processor_name)

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (ROOT_PATH / (arguments['--output-dir'])).resolve() if arguments['--output-dir'] else (input_path / "input")
output_name = arguments['--output-name'] if arguments['--output-name'] else "data.csv"
data_path = (ROOT_PATH / (arguments['--data-dir'])).resolve() if arguments['--data-dir'] else processor.get_default_data_path(input_path)

data_set = arguments['--data-set'] if arguments['--data-set'] else (input_path).resolve().name
frame_rate = int(arguments['--frame-rate']) if arguments['--frame-rate'] else 180

tiff_path = (ROOT_PATH / (arguments['--img-dir'])).resolve() if arguments['--img-dir'] else (input_path / ("images/" + data_set))

filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 1
pixel_size = float(arguments['--pixel-size']) if arguments['--pixel-size'] else 1
rolling_ball_size = int(arguments['--rolling-ball-size']) if arguments['--rolling-ball-size'] else 30


### Preprocess our data
params = {
  'data_set': data_set,
  'input_path': input_path,
  'tiff_path': tiff_path,
  'frame_rate': frame_rate,
  'filter_window': filter_window,
  'gamma': gamma,
  'channel': channel,
  'pixel_size': pixel_size,
  'rolling_ball_size': rolling_ball_size
}

data = processor.process_data(data_path, params)

output_path.mkdir(exist_ok=True)
output_file_path = (output_path / (output_name)).resolve()

data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

exit()