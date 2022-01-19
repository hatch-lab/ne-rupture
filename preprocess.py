# coding=utf-8

"""
Gets data into a format readable by classifiers

Usage:
  preprocess.py [options] PROCESSOR INPUT [<args>...]

Arguments:
  PROCESSOR The kind of image processor to use (eg, imaris or matlab)
  INPUT Path to the directory containing the raw data (CSV files for Imaris, TIFFs for MATLAB)

Options:
  -h --help Show this screen.
  --output-dir=<string>  [default: input] The subdirectory to save the resulting CSV file
  --output-name=<string>  [default: data.csv] The name of the resulting CSV file
  --img-dir=<string>  [defaults: INPUT/images/(data_set)] The path to TIFF files
  --data-dir=<string>  Where to find the raw data. Typically determined by the preprocessor you've selected.
  --channel=<int>  [default: 1] The channel to keep (ie, the NLS-3xGFP channel)
  --data-set=<string>  The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<float>  [default: 1] Pixels per micron. If 0, will attempt to detect automatically.
  --frame-rate=<int>  [default: 180] The seconds that elapse between frames
  --remake-tracks  [default: False] Whether to skip processing all data from scratch and just rebuild tracks/modify filters

Output:
  A CSV file with processed data
"""

import sys
import os
from pathlib import Path
from importlib import import_module

ROOT_PATH = Path(__file__ + "/..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
from lib.preprocessor import base_transform
from lib.tracks import make_tracks

import lib.video as hatchvid
import pandas as pd

import re

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

arguments = docopt(__doc__, version=get_version(), options_first=True)

processor_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['PROCESSOR'])
if processor_name != arguments['PROCESSOR']:
  print(colorize("yellow", "Processor has been sanitized to " + processor_name))

if len(processor_name) > 0 and os.path.sep not in processor_name and ((ROOT_PATH / ('preprocessors/' + str(processor_name) + '.py'))).is_file():
  processor = import_module("preprocessors." + processor_name)
else:
  raise Exception('That preprocessor does not exist.')

processor_arguments = docopt(processor.__doc__, argv=[arguments['PROCESSOR']] + [arguments['INPUT']] + arguments['<args>'])
processor_schema = processor.get_schema()

arguments.update(processor_arguments)

schema = {
  'PROCESSOR': And(len, lambda n: (ROOT_PATH / ('preprocessors/' + str(n) + '.py')).is_file(), error='That preprocessor does not exist'),
  'INPUT': And(len, lambda n: os.path.exists(n), error='That folder does not exist'),
  '--output-dir': len,
  '--output-name': len,
  '--img-dir': Or(None, len),
  '--data-dir': Or(None, len),
  '--channel': And(Use(int), lambda n: n > 0, error='--channel must be > 0'),
  '--data-set': Or(None, len),
  '--pixel-size': And(Use(float), lambda n: n >= 0, error='--pixel-size must be > 0'),
  '--frame-rate': And(Use(int), lambda n: n > 0),
  '--help': Or(None, bool),
  Optional('--remake-tracks'): bool,
  '--version': Or(None, bool),
  Optional('<args>'): lambda n: True,
  Optional('--'): lambda n: True
}
schema.update(processor_schema)

try:
  arguments = Schema(schema).validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

### Arguments and inputs
input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / (arguments['--output-dir']))
data_path = (ROOT_PATH / (arguments['--data-dir'])).resolve() if arguments['--data-dir'] else processor.get_default_data_path(input_path)

arguments['--data-set'] = arguments['--data-set'] if arguments['--data-set'] else (input_path).name

tiff_path = (input_path / (arguments['--img-dir'])).resolve() if arguments['--img-dir'] else (input_path / ("images/" + arguments['--data-set']))

arguments['input_path'] = input_path
arguments['output_path'] = output_path
arguments['data_path'] = data_path
arguments['tiff_path'] = tiff_path

arguments['--pixel-size'] = arguments['--pixel-size'] if arguments['--pixel-size'] > 0 else None

### Preprocess our data
output_path.mkdir(exist_ok=True)
output_file_path = (output_path / (arguments['--output-name'])).resolve()
if not arguments['--remake-tracks'] or not output_file_path.exists() or not output_file_path.is_file():
  data = processor.process_data(data_path, arguments)
  data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)
else:
  data = pd.read_csv(str(output_file_path, dtype={ 'particle_id': str, 'orig_particle_id': str }))
  data['particle_id'] = data['orig_particle_id']

### Launch GUI for assigning tracks and filtering

# Assign particles to tracks
print("Building tracks...")
frames = sorted(data['frame'].unique())
data['orig_particle_id'] = data['particle_id']

data['min_frame'] = data['frame']
for i in tqdm(frames[:-1], unit=" frames"):
  data = make_tracks(data, i, 10, 3)
data.drop_duplicates(subset=[ 'particle_id', 'frame' ], inplace=True)
data.drop('min_frame', axis='columns', inplace=True)

# # "Fill in" gaps where we lost tracking
# data.sort_values(by=[ 'particle_id', 'frame' ], inplace=True)
# data['track_id'] = 0
# data = data.groupby([ 'particle_id' ]).apply(id_track)
# missing_particle_ids = data.loc[( data['track_id'] > 0 ), 'particle_id'].unique()

params['frame_width'] = frame_shape[1]
params['frame_height'] = frame_shape[0]
data = base_transform(data, params)
data = pd.read_csv(str(output_file_path), dtype={ 'particle_id': str })

### Set up filtering
video_path = output_path / "videos"
video_path.mkdir(mode=0o755, parents=True, exist_ok=True)

data.loc[:,'frame_path'] = data.apply(lambda x: 
  str( (tiff_path / (str(x.frame).zfill(4) + '.tif')).resolve() ), axis=1
)
data_set = data['data_set'].unique()[0]
hatchvid.make_video(data, video_path / (data_set + ".mp4"), annotate=True, draw_tracks=True, movie_name=data_set)