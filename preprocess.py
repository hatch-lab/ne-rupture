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
  --min-track-length=<int>  [default: 5] Any tracks with fewer than these frames will be filtered out. The minimum track length must always be at least 4, in order to generate derivatives. Set to 0 to skip any track filtering.
  --edge-filter=<int>  [default: 50] Filters cells that are near the edge of the frame, in pixels.
  --accept-tracks  [default: False] Whether to just accept the tracks; skip asking to check
  --skip-tracks  [default: False] Treat each frame as a separate position; don’t draw tracks or calculate some derived features, which would require a timeline
  --skip-flat-field-correction  [default: False] Don't perform psuedo-flat field correction

Output:
  A CSV file with processed data
"""

import sys
import os
import shutil
from pathlib import Path, PurePath
from importlib import import_module
import builtins
import cv2

ROOT_PATH = Path(__file__ + "/..").resolve()
builtins.ROOT_PATH = ROOT_PATH

sys.path.append(str(ROOT_PATH))

from docopt import docopt
from lib.version import get_version
from lib.output import colorize
from lib.preprocessor import base_transform, make_tracks, open_file
import lib.video as hatchvid

from skimage import measure, morphology, exposure, filters, restoration, registration, color

import defusedxml.ElementTree as ET
import math
import numpy as np
import pandas as pd
import tifffile
import json
import copy
from time import sleep

from yaspin import yaspin
from yaspin.spinners import Spinners

import re

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

from inputs.DirectoryInput import DirectoryInput
from inputs.NDInput import NDInput

arguments = docopt(__doc__, version=get_version(), options_first=True)

processor_name = re.sub(r'[^a-zA-Z0-9\-\_\.\+]', '', arguments['PROCESSOR'])
if processor_name != arguments['PROCESSOR']:
  print(colorize("yellow", "Processor has been sanitized to " + processor_name))
  sleep(3)

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
  Optional('--skip-flat-field-correction'): bool,
  Optional('<args>'): lambda n: True,
  Optional('--'): lambda n: True,
  Optional('--accept-tracks'): bool,
  Optional('--skip-tracks'): bool,
}
schema.update(processor_schema)

try:
  arguments = Schema(schema).validate(arguments)
except SchemaError as error:
  print(colorize("red", str(error)))
  exit(1)

### Arguments and inputs
input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / (arguments['--output-dir']))
output_path.mkdir(exist_ok=True, mode=0o755)
data_path = (input_path / (arguments['--data-dir'])).resolve() if arguments['--data-dir'] else processor.get_default_data_path(input_path)
arguments['input_path'] = input_path
arguments['output_path'] = output_path

## MAKE THESE INTO REAL ARGS
calc_optical_flow = False

# Get TIFF stacks
if data_path.suffix == ".nd":
  input_gen = NDInput(input_path, data_path, arguments['--channel'])
else:
  input_gen = DirectoryInput(input_path, data_path, arguments['--channel'])
arguments['--data-set'] = [ arguments['--data-set'] ] if arguments['--data-set'] else input_gen.get_data_sets()

arguments['pixel_size'] = arguments['--pixel-size'] if arguments['--pixel-size'] > 0 else None
if arguments['pixel_size'] is None:
  arguments['pixel_size'] = input_gen.get_spatial_calibration()

if arguments['pixel_size'] is None:
  # We need pixel size specified
  arguments['pixel_size'] = float(input("Microns/pixel could not be determined from the TIFFs.\nCommon values are:\nLeica SD 40x: " + str(0.2538) + "\nLeica SD 20x: " + str(0.5089) + "\nPlease enter a spatial calibration value (um/pixel):"))

datas = []
for data_set in arguments['--data-set']:
  print("Processing {}".format(data_set))

  stack = input_gen.get_stack(data_set, skip_flat_field=True, skip_rolling_ball=True)

  if len(stack) <= arguments['--gap-size'] and arguments['--skip-tracks'] is False:
    print(colorize("red", "--gap-size must be less than the total number of frames"))
    exit(1)

  if len(stack) <= arguments['--min-track-length'] and arguments['--skip-tracks'] is False:
    print(colorize("yellow", "--min-track-length is longer than the number of frames. No track filtering will be performed."))
    arguments['--min-track-length'] = 0
    sleep(3)  

  data_set_path = (input_path / (arguments['--img-dir'])).resolve() if arguments['--img-dir'] else (input_path / ("images/" + data_set))
  data_set_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  normalized_path = data_set_path / "normalized"
  normalized_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  masks_path = data_set_path / "masks"
  masks_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  ### Segment our data
  processor.segment(stack[...,input_gen.get_segmentation_channel_idx()], normalized_path, masks_path, pixel_size=arguments['pixel_size'], params=arguments)

  ### Assign particles to tracks
  tracks_path = data_set_path / "tracks"
  tracks_path.mkdir(exist_ok=True, mode=0o755)

  cyto_tracks_path = data_set_path / "cyto-tracks"
  cyto_tracks_path.mkdir(exist_ok=True, mode=0o755)
  
  gap_size = arguments['--gap-size']
  roi_size = arguments['--roi-size']

  preview_video_path = data_set_path / 'tracks.mp4'
  print("Building tracks with gap size {:d} and roi size {:f}...".format(gap_size, roi_size))
  tracks = make_tracks(normalized_path, masks_path, tracks_path, cyto_tracks_path, delta_t=arguments['--gap-size'], default_roi_size=arguments['--roi-size'])
  
  d = processor.extract_features(stack, tracks_path, cyto_tracks_path, input_gen.get_channels(), pixel_size=arguments['pixel_size'], params=arguments)
  d['data_set'] = data_set
  datas.append(d)

  hatchvid.make_video(stack[...,input_gen.get_segmentation_channel_idx()], tracks_path, data_set_path / 'video.mp4')
data = pd.concat(datas)
arguments['frame_width'] = stack.shape[2]
arguments['frame_height'] = stack.shape[1]

if arguments['--skip-tracks']:
  data['particle_id'] = data['frame'].astype(str) + '.' + data['particle_id']
else:
  data = base_transform(stack.shape[3], data, arguments)

output_file_path = (output_path / (arguments['--output-name'])).resolve()
data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

json_path = output_path / "preprocess.conf.json"
print("Saving configration options to " + str(json_path))
with open(str(json_path), 'w') as fp:
  json_arguments = copy.deepcopy(arguments)
  for key,arg in arguments.items():
    if isinstance(arg, PurePath):
      json_arguments[key] = str(arg)
  fp.write(json.dumps(json_arguments))

print("Finished!")

  # arguments['data_path'] = data_path
  # arguments['tiff_path'] = tiff_path
  

  # ### Extract TIFFs to make single channel, single frame TIFFs
  # tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    
  # extracted_path = tiff_path / "corrected"
  # extracted_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # segmentation_path = extracted_path / ("channel_" + str(arguments['--channel']))

  # optical_flow_path = tiff_path / "optical-flow"
  # optical_flow_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # masks_path = tiff_path / "masks"
  # masks_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # # Frame data to store
  # frame_shape = None
  # if arguments['pixel_size'] is None:
  #   arguments['pixel_size'] = input_gen.get_spatial_calibration()

  # if arguments['pixel_size'] is None:
  #   # We need pixel size specified
  #   arguments['pixel_size'] = float(input("Microns/pixel could not be determined from the TIFFs.\nCommon values are:\nLeica SD 40x: " + str(0.2538) + "\nLeica SD 20x: " + str(0.5089) + "\nPlease enter a spatial calibration value (um/pixel):"))

  # with yaspin(text="Extracting individual TIFFs") as spinner:
  #   spinner.spinner = Spinners.dots8
  #   for channel in input_gen.get_channels():
  #     stack = []
  #     for file in input_gen.get_files(data_set, channel=channel):
  #       with tifffile.TiffFile(file) as tif:
  #         for i in range(len(tif.pages)):
  #           file_name = str(len(stack)+1).zfill(4) + ".tif"
  #           spinner.text = "Extracting {name}...".format(name=file_name)
  #           img = tif.pages[i].asarray()
  #           if frame_shape is None:
  #             frame_shape = img.shape

  #           if not arguments['--skip-flat-field-correction']:
  #             img = exposure.rescale_intensity(img, out_range=(0, 1)).astype(np.float64)

  #             # Generate pseudo-flat field correction 
  #             pseudo_flat_image = filters.gaussian(img, sigma=300, preserve_range=False)
  #             # tifffile.TiffWriter(str(extracted_path / ("pseudo" + str(len(stack)) + ".tif"))).write(pseudo_flat_image, resolution=(1/arguments['pixel_size'], 1/arguments['pixel_size'], None))
  #             img /= pseudo_flat_image * np.mean(pseudo_flat_image)
  #             img = exposure.rescale_intensity(img, out_range=(0,255))
  #             img[img > 255] = 255

  #             # img -= restoration.rolling_ball(img, radius=100)
  #           else:
  #             img = exposure.rescale_intensity(img, out_range=(0,255))
            
  #           img = img.astype(np.uint8)
  #           stack.append(img)

  #     spinner.text = "Performing rolling ball background subtraction..."
  #     # bg = restoration.rolling_ball(stack, kernel=restoration.ellipsoid_kernel((2, 100, 100), 100))
  #     stack = np.stack(stack, axis=0)
  #     # stack -= bg

  #     spinner.text = "Calculating optical flow..."
  #     previous_img = None
  #     flows = []
  #     for idx,img in enumerate(stack):
  #       file_name = str((idx+1)).zfill(4) + ".tif"
  #       tifffile.TiffWriter(str(extracted_path / ("channel_" + str(channel) + "/" + file_name))).write(img, resolution=(1/arguments['pixel_size'], 1/arguments['pixel_size'], None))
  #       # if previous_img is not None:
  #       #   flow = cv2.calcOpticalFlowFarneback(previous_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  #       #   mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  #       #   hsv = np.zeros(( ang.shape[0], ang.shape[1], 3))
  #       #   hsv[..., 1] = 255
  #       #   hsv[..., 0] = ang*180/np.pi/2
  #       #   hsv[..., 2] = mag #cv2.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
  #       #   flows.append(hsv)
  #       # previous_img = img.copy()

  #     # flows = np.array(flows)
  #     # flows = exposure.rescale_intensity(flows, out_range=np.uint32)
  #     # for idx,img in enumerate(flows):
  #     #   file_name = str((idx+2)).zfill(4) + ".tif"
  #     #   rgb = color.hsv2rgb(img)
  #     #   tifffile.TiffWriter(str(optical_flow_path / file_name)).write(rgb, resolution=(1/arguments['pixel_size'], 1/arguments['pixel_size'], None))

  #     spinner.write("Found " + str(len(stack)) + " images")
  #   spinner.ok("✅")

  # if len(stack) <= arguments['--gap-size'] and arguments['--skip-tracks'] is False:
  #   print(colorize("red", "--gap-size must be less than the total number of frames"))
  #   exit(1)

  # if len(stack) <= arguments['--min-track-length'] and arguments['--skip-tracks'] is False:
  #   print(colorize("yellow", "--min-track-length is longer than the number of frames. No track filtering will be performed."))
  #   arguments['--min-track-length'] = 0
  #   sleep(3)

  # ### Segment our data
  # # processor.segment(data_path, tiff_path, segmentation_path, masks_path, pixel_size=arguments['pixel_size'], channel=arguments['--channel'], params=arguments)

  # ### Assign particles to tracks
  # build_tracks = (arguments['--skip-tracks'] is False)
  # tracks_path = tiff_path / "tracks"
  # tracks_path.mkdir(exist_ok=True, mode=0o755)
  # gap_size = arguments['--gap-size']
  # roi_size = arguments['--roi-size']

  # (ROOT_PATH / 'tmp').mkdir(exist_ok=True, mode=0o755)
  # preview_video_path = ROOT_PATH / 'tmp/current_tracks.mp4'

  # if len(list(masks_path.glob('*.tif'))) <= 0:
  #   # hatchvid.make_video(tiff_path / "corrected", tracks_path, tiff_path / 'video.mp4')
  #   build_tracks = False

  # if len(list((tiff_path / "cyto-tracks").glob('*.tif'))) > 0:
  #   continue

  # while build_tracks:
  #   print("Building tracks with gap size {:d} and roi size {:f}...".format(gap_size, roi_size))
  #   make_tracks(tiff_path, tracks_path, delta_t=arguments['--gap-size'], default_roi_size=arguments['--roi-size'])

  #   ### Display tracks
  #   show_gui = (arguments['--accept-tracks'] is False)

  #   if show_gui:
  #     hatchvid.make_video(extracted_path, tracks_path, preview_video_path)
      
  #     print('Opening a preview of the tracks. Indicate if you like them or want to change the parameters.')
  #     sleep(2)

  #     open_file(str(preview_video_path))

  #     change = input('Do you like the tracks? [Y/n]')

  #     if change.upper() != 'Y':
  #       new_gap_size = int(input("Enter a new gap size ({:d}): ".format(gap_size)))
  #       new_roi_size = float(input("Enter a new ROI size ({:f}): ".format(roi_size)))

  #       if new_gap_size == '':
  #         new_gap_size = gap_size

  #       if new_roi_size == '':
  #         new_roi_size = roi_size

  #       if new_gap_size == gap_size and new_roi_size == roi_size:
  #         build_tracks = False
  #       else:
  #         if new_gap_size > 0:
  #           gap_size = new_gap_size
  #         if new_roi_size >= 1:
  #           roi_size = new_roi_size
  #     else:
  #       build_tracks = False
  #   else:
  #     build_tracks = False

  # # Clean up
  # if preview_video_path.exists():
  #   preview_video_path.unlink()

  # ### Extract features
  # if arguments['--skip-tracks']:
  #   # Move masks to tracks
  #   mask_files = masks_path.glob("*.tif")
  #   for mask_file in mask_files:
  #     if (tracks_path / mask_file.name).exists():
  #       (tracks_path / mask_file.name).unlink()
        
  #     mask_file.rename((tracks_path / mask_file.name))

  # cyto_tracks_path = tiff_path / "cyto-tracks"
  # cyto_tracks_path.mkdir(exist_ok=True, mode=0o755)
  # d = processor.extract_features(tiff_path, tracks_path, cyto_tracks_path, input_gen.get_channels(), pixel_size=arguments['pixel_size'], params=arguments)
  # continue
  # d['data_set'] = data_set
  # datas.append(d)

  # print("Removing temporary files...")
  # try:
  #   # shutil.rmtree(str(extracted_path))
  #   shutil.rmtree(str(masks_path))
  # except:
  #   print(colorize("yellow"), "Unable to remove temporary files in:\n" + str(extracted_path) + "\n" + str(masks_path))

  # arguments['--accept-tracks'] = True # Tracks have already been accepted

# data = pd.concat(datas)
# arguments['frame_width'] = frame_shape[1]
# arguments['frame_height'] = frame_shape[0]

# if arguments['--skip-tracks']:
#   data['particle_id'] = data['frame'].astype(str) + '.' + data['particle_id']
# else:
#   data = base_transform(data, arguments)

# output_file_path = (output_path / (arguments['--output-name'])).resolve()
# data.to_csv(str(output_file_path), header=True, encoding='utf-8', index=None)

# json_path = output_path / "preprocess.conf.json"
# print("Saving configration options to " + str(json_path))
# with open(str(json_path), 'w') as fp:
#   json_arguments = copy.deepcopy(arguments)
#   for key,arg in arguments.items():
#     if isinstance(arg, PurePath):
#       json_arguments[key] = str(arg)
#   fp.write(json.dumps(json_arguments))

# print("Finished!")
