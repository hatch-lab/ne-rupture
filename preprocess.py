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
  --pixel-size=<float>  [default: 0] Pixels per micron. If 0, will attempt to detect automatically.
  --frame-rate=<int>  [default: 180] The seconds that elapse between frames
  --gap-size=<int>  [default: 5] The maximum gap size when building tracks
  --roi-size=<float>  [default: 2.0] Given a segment at time t+1, will search for a shape to match one found at time t. The search distance is the median shape size*roi-size
  --min-track-length=<int>  [default: 5] Any tracks with fewer than these frames will be filtered out. The minimum track length must always be at least 4, in order to generate derivatives.
  --edge-filter=<int>  [default: 50] Filters cells that are near the edge of the frame, in pixels.

Output:
  A CSV file with processed data
"""

import sys
import os
import shutil
from pathlib import Path
from importlib import import_module
import builtins

ROOT_PATH = Path(__file__ + "/..").resolve()
builtins.ROOT_PATH = ROOT_PATH

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
from lib.preprocessor import base_transform, make_tracks, open_file
import lib.video as hatchvid

import defusedxml.ElementTree as ET
import math
import numpy as np
import pandas as pd
import tifffile

from yaspin import yaspin
from yaspin.spinners import Spinners

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
  '--roi-size': And(Use(float), lambda n: n >= 1, error='--roi-size must be >= 1.0'),
  '--min-track-length': And(Use(int), lambda n: n >= 4, error='--min-track-length must be >= 4 frames'),
  '--edge-filter': And(Use(int), lambda n: n >= 0, error='--edge-filter must be >= 0 px'),
  Optional('<args>'): lambda n: True,
  Optional('--'): lambda n: True
}
schema.update(processor_schema)

try:
  arguments = Schema(schema).validate(arguments)
except SchemaError as error:
  print(colorize("red", error))
  exit(1)

### Arguments and inputs
input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / (arguments['--output-dir']))
output_path.mkdir(exist_ok=True, mode=0o755)
data_path = (ROOT_PATH / (arguments['--data-dir'])).resolve() if arguments['--data-dir'] else processor.get_default_data_path(input_path)

arguments['--data-set'] = arguments['--data-set'] if arguments['--data-set'] else (input_path).name

tiff_path = (input_path / (arguments['--img-dir'])).resolve() if arguments['--img-dir'] else (input_path / ("images/" + arguments['--data-set']))

arguments['input_path'] = input_path
arguments['output_path'] = output_path
arguments['data_path'] = data_path
arguments['tiff_path'] = tiff_path

arguments['--pixel-size'] = arguments['--pixel-size'] if arguments['--pixel-size'] > 0 else None

### Extract TIFFs to make single channel, single frame TIFFs
tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)
  
extracted_path = tiff_path / "extracted"
extracted_path.mkdir(mode=0o755, parents=True, exist_ok=True)

masks_path = tiff_path / "masks"
masks_path.mkdir(mode=0o755, parents=True, exist_ok=True)

# Get TIFF stacks
files = list(data_path.glob("*.tif")) + list(data_path.glob("*.TIF")) + list(data_path.glob("*.tiff")) + list(data_path.glob("*.TIFF"))
files = list(filter(lambda x: x.name[:2] != "._", files))

if len(files) <= 0:
  raise  NoImagesFound()

files.sort(key=lambda x: str(len(str(x))) + str(x).lower())
files = sorted(set(files), key=lambda x: str(len(str(x))) + str(x).lower())

# Frame data to store
frame_shape = None
frame_i = 1

if arguments['--pixel-size'] is None:
  with tifffile.TiffFile(files[0]) as tif:
    if 'spatial-calibration-x' in tif.pages[0].description:
      # Try from the description

      metadata = ET.fromstring(tif.pages[0].description)
      plane_data = metadata.find("PlaneInfo")

      for prop in plane_data.findall("prop"):
        if prop.get("id") == "spatial-calibration-x":
          arguments['--pixel-size'] = float(prop.get("value"))
          break
    
    elif 'XResolution' in tif.pages[0].tags:
      # Try from the XResolution tag
      arguments['--pixel-size'] = tif.pages[0].tags['XResolution'].value

      if len(arguments['--pixel-size']) == 2:
        arguments['--pixel-size'] = arguments['--pixel-size'][0]/arguments['--pixel-size'][1]

      arguments['--pixel-size'] = 1/arguments['--pixel-size']

if arguments['--pixel-size'] is None:
  # We need pixel size specified
  arguments['--pixel-size'] = float(input("Microns/pixel could not be determined from the TIFFs.\nCommon values are:\nLeica SD 40x: " + str(0.2538) + "\nLeica SD 20x: " + str(0.5089) + "\nPlease enter a spatial calibration value (um/pixel):"))

with yaspin(text="Extracting individual TIFFs") as spinner:
  spinner.spinner = Spinners.dots8
  for file in files:
    with tifffile.TiffFile(file) as tif:
      for i in range(len(tif.pages)):
        img = tif.pages[i].asarray()

        # Get the signal channel
        if len(img.shape) == 3:
          # channel is 1-indexed, python is 0-indexed
          img = img[:,:, (channel-1)]

        if frame_shape is None:
          frame_shape = img.shape

        file_name = str(frame_i).zfill(4) + ".tif"
        tifffile.TiffWriter(str(extracted_path / file_name)).save(img, resolution=(1/arguments['--pixel-size'], 1/arguments['--pixel-size'], None))
        frame_i += 1
  spinner.write("Found " + str(frame_i-1) + " images")
  spinner.ok("✅")

if frame_i-1 < arguments['--gap-size']:
  print(colorize("red", "--gap-size must be less than the total number of frames"))
  exit(1)

### Segment our data
tiff_path.mkdir(exist_ok=True, mode=0o755)
processor.segment(data_path, tiff_path, extracted_path, masks_path, pixel_size=arguments['--pixel-size'], channel=arguments['--channel'], params=arguments)

### Assign particles to tracks
build_tracks = True
tracks_path = tiff_path / "tracks"
tracks_path.mkdir(exist_ok=True, mode=0o755)
gap_size = arguments['--gap-size']
roi_size = arguments['--roi-size']

(ROOT_PATH / 'tmp').mkdir(exist_ok=True, mode=0o755)
preview_video_path = ROOT_PATH / 'tmp/current_tracks.mp4'

while build_tracks:
  print("Building tracks with gap size {:d} and roi size {:f}...".format(gap_size, roi_size))
  make_tracks(tiff_path, tracks_path, delta_t=arguments['--gap-size'], default_roi_size=arguments['--roi-size'])

  ### Display tracks
  show_gui = True

  if show_gui:
    hatchvid.make_video(tiff_path, tracks_path, preview_video_path)
    
    print('Opening a preview of the tracks. Indicate if you like them or want to change the parameters.')

    open_file(str(preview_video_path))

    change = input('Do you like the tracks? [Y/n]')

    if change.upper() != 'Y':
      new_gap_size = int(input("Enter a new gap size ({:d}): ".format(gap_size)))
      new_roi_size = float(input("Enter a new ROI size ({:f}): ".format(roi_size)))

      if new_gap_size == '':
        new_gap_size = gap_size

      if new_roi_size == '':
        new_roi_size = roi_size

      if new_gap_size == gap_size and new_roi_size == roi_size:
        build_tracks = False
      else:
        if new_gap_size > 0:
          gap_size = new_gap_size
        if new_roi_size >= 1:
          roi_size = new_roi_size
    else:
      build_tracks = False

# Clean up
if preview_video_path.exists():
  preview_video_path.unlink()

### Extract features
cyto_tracks_path = tiff_path / "cyto-tracks"
cyto_tracks_path.mkdir(exist_ok=True, mode=0o755)
data = processor.extract_features(tiff_path, tracks_path, cyto_tracks_path, pixel_size=arguments['--pixel-size'], params=arguments)

arguments['frame_width'] = frame_shape[1]
arguments['frame_height'] = frame_shape[0]

data = base_transform(data, arguments)

output_file_path = (output_path / (arguments['--output-name'])).resolve()
data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

print("Removing temporary files...")
shutil.rmtree(str(extracted_path))
shutil.rmtree(str(masks_path))
print("Finished!")
