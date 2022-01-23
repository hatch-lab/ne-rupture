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
  --gap-size=<int>  [default: 3] The maximum gap size when building tracks
  --roi-size=<float>  [default: 2.0] Given a segment at time t+1, will search for a shape to match one found at time t. The search distance is the median shape size*roi-size

Output:
  A CSV file with processed data
"""

import sys
import os
from pathlib import Path
from importlib import import_module

ROOT_PATH = Path(__file__ + "/..").resolve()

FONT_PATH = (ROOT_PATH / ("lib/fonts/font.ttf")).resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
from lib.preprocessor import base_transform, make_tracks
# from lib.tracks import make_tracks

import math
import lib.video as hatchvid
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.color import label2rgb
from skimage.measure import regionprops_table
from tqdm import tqdm
import subprocess

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
  '--version': Or(None, bool),
  '--gap-size': And(Use(int), lambda n: n > 0, error='--gap-size must be > 0'),
  '--roi-size': And(Use(float), lambda n: n >= 1, error='--roi-size must be >= 1'),
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

### Segment our data
tiff_path.mkdir(exist_ok=True, mode=0o755)
frame_info = processor.segment(data_path, tiff_path, pixel_size=arguments['--pixel-size'], params=arguments)

### Assign particles to tracks
build_tracks = True
tracks_path = tiff_path / "tracks"
tracks_path.mkdir(exist_ok=True, mode=0o755)
gap_size = arguments['--gap-size']
roi_size = arguments['--roi-size']

lut = pd.read_csv(str(ROOT_PATH / "lib/luts/glasbey.lut"), dtype={'red': int, 'green': int, 'blue': int})
# label2rgb expects colors in a [0,1] range
lut['red_float'] = lut['red']/255
lut['green_float'] = lut['green']/255
lut['blue_float'] = lut['blue']/255

while build_tracks:
  print("Building tracks with gap size {:d} and roi size {:f}...".format(gap_size, roi_size))
  make_tracks(tiff_path, tracks_path, delta_t=arguments['--gap-size'], default_roi_size=arguments['--roi-size'])

  ### Display tracks
  show_gui = True

  if show_gui:
    # Annotate frames
    image_files = list(tiff_path.glob("*.tif"))
    image_files.sort(key=lambda x: str(len(str(x))) + str(x))
    annotated_frames = []

    # Build track
    track_frame = None
    old_props = None
    top_padding = 25
    for image_file in tqdm(image_files, desc='Generating preview'):
      track_file = tracks_path / image_file.name
      frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
      mask = cv2.imread(str(track_file), cv2.IMREAD_ANYDEPTH)
      if track_frame is None:
        track_frame = Image.new('RGBA', ( frame.shape[1], frame.shape[0] ), (0,0,0,0))
        track_draw = ImageDraw.Draw(track_frame)

      props = pd.DataFrame(regionprops_table(mask, properties=('label', 'centroid')))
      props.rename(columns={ 'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
      if props.shape[0] > lut.shape[0]:
        new_luts = lut.sample(props.shape[0]-lut.shape[0], replace=True)
        lut = pd.concat(lut, new_luts)
      
      props = props.merge(lut, left_on='label', right_index=True)

      r = props['red_float'].tolist()
      g = props['green_float'].tolist()
      b = props['blue_float'].tolist()
      colors = list(zip(r, g, b))

      frame = label2rgb(mask, image=frame, colors=colors, alpha=0.4)

      if old_props is not None:
        old_props.rename(columns={ 'x': 'old_x', 'y': 'old_y' }, inplace=True)
        coords = props.merge(old_props, on='label')
        for row in coords.itertuples():
          track_draw.line([ row.old_x, row.old_y, row.x, row.y ], fill=(row.red, row.green, row.blue), width=1)
      old_props = props[['x', 'y', 'label']]

      # Add padding for text
      frame = Image.fromarray((frame*255).astype(np.uint8))
      new_frame = Image.new('RGBA', (frame.size[0], frame.size[1]+top_padding), (0,0,0,0))
      new_frame.paste(frame, ( 0, top_padding ))
      frame = new_frame

      font_color = 'rgb(255,255,255)'
      small_font = ImageFont.truetype(str(FONT_PATH), size=14)
      draw = ImageDraw.Draw(frame)

      frame_idx = int(track_file.stem)
      time = frame_idx*arguments['--frame-rate']

      hours = math.floor(time / 3600)
      minutes = math.floor((time - (hours*3600)) / 60)
      seconds = math.floor((time - (hours*3600)) % 60)

      label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, frame_idx)
      draw.text((10, 10), label, fill=font_color, font=small_font)

      # Add progress bar
      width = int(frame_idx/len(image_files)*frame.size[0])
      draw.rectangle([ (0, 0), (width, 5) ], fill=font_color)

      annotated_frames.append(frame)

    # Add in track frame and write out movie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(ROOT_PATH / 'tmp/current_tracks.mp4'), fourcc, 10, (annotated_frames[0].size[0], annotated_frames[0].size[1]), True)
    for annotated_frame in annotated_frames:
      annotated_frame.paste(track_frame, ( 0, top_padding ), track_frame)
      annotated_frame = np.asarray(annotated_frame.convert('RGB'))
      writer.write(annotated_frame)
    writer.release()

    print('Opening a preview of the tracks. Indicate if you like them or want to change the parameters.')

    cmd = [
      'open',
      str(ROOT_PATH / 'tmp/current_tracks.mp4')
    ]
    subprocess.call(cmd)

    change = input('Do you like the tracks? [Y/n]')

    if change.upper() != 'Y':
      new_gap_size = int(input("Enter a new gap size ({:d}): ".format(gap_size)))
      new_roi_size = float(input("Enter a new ROI size ({:f}): ".format(roi_size)))

      if new_gap_size > 0:
        gap_size = new_gap_size
      if new_roi_size >= 1:
        roi_size = new_roi_size
    else:
      build_tracks = False

# Clean up
if (ROOT_PATH / 'tmp/current_tracks.mp4').exists():
  (ROOT_PATH / 'tmp/current_tracks.mp4').unlink()

### Extract features
data = processor.extract_features(tiff_path, pixel_size=arguments['--pixel-size'], params=arguments)
output_file_path = (output_path / (arguments['--output-name'])).resolve()
data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

arguments['frame_width'] = frame_info['frame_shape'][1]
arguments['frame_height'] = frame_info['frame_shape'][0]

data = base_transform(data, arguments)

output_file_path = (output_path / (arguments['--output-name'])).resolve()
data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)
