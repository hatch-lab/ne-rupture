# coding=utf-8

"""
Gets data into a format readable by classifiers, using cellprofiler for segmentation and feature extraction

Usage:
  preprocess.py cellprofiler [options] [--] INPUT

Arguments:
  INPUT Path to the directory containing the raw data (CSV files for Imaris, TIFFs for MATLAB)

Options:
  -h, --help
  -v, --version
  --pipeline-path=<string>  [default: preprocessors/cellprofiler/default.cppipe] The CellProfiler pipelline to use
"""

import sys
import os
from pathlib import Path
import shutil

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

from docopt import docopt
from lib.output import colorize

from errors.NoImagesFound import NoImagesFound

import numpy as np
import pandas as pd
import subprocess

import tifffile
from skimage import filters, morphology
import scipy 

from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

NAME      = "cellprofiler"


def get_schema():
  return {
  'cellprofiler': lambda x: True,
    '--pipeline-path': Or(None, lambda n: (ROOT_PATH / n).is_file(), error='That pipeline does not exist')
  }

def get_default_data_path(input_path):
  """
  Returns the path to raw data
  
  Arguments:
    input_path str The base input path

  Returns
    str The path to raw data
  """
  return input_path / "images/raw"

def process_data(data_path, params):
  """
  Process raw data, segment it, and extract features

  Arguments:
    data_path str The path to raw data
    params dict A dictionary of parameters

  Return:
    pandas.DataFrame The extracted features for each cell
  """

  data_set = params['--data-set']
  input_path = params['input_path']
  channel = params['--channel']
  pixel_size = params['--pixel-size']
  tiff_path = params['tiff_path']
  pipeline_path = Path(params['--pipeline-path']).resolve() if params['--pipeline-path'] is not None else ROOT_PATH / ('preprocessors/cellprofiler/default.cppipeline')
  keep_imgs = params['--keep-imgs']

  tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)
  
  raw_path = tiff_path / "singletons"
  raw_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  mask_path = tiff_path / "masks"
  mask_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # Get TIFF stacks
  files = list(data_path.glob("*.tif"))
  if len(files) <= 0:
    # TODO: Make this...not crappy
    files = list(data_path.glob("*.TIF"))
  # Filter out ._ files OS X likes to make
  files = list(filter(lambda x: x.name[:2] != "._", files))

  if len(files) <= 0:
    raise  NoImagesFound()
  files = [ str(x) for x in files ]
  files.sort(key=lambda x: str(len(x)) + x)  

  # Frame data to store
  frame_shape = None
  frame_i = 1

  with yaspin(text="Extracting individual TIFFs") as spinner:
    spinner.spinner = Spinners.dots8
    for file in files:
      with tifffile.TiffFile(file) as tif:

        if pixel_size is None and 'XResolution' in tif.pages[0].tags:
          pixel_size = tif.pages[0].tags['XResolution'].value
          dtype = tif.pages[0].tags['XResolution'].dtype

          if len(pixel_size) == 2:
            pixel_size = pixel_size[0]

          if dtype == '1I':
            # Convert from inches to microns
            pixel_size = pixel_size*3.937E-5
          elif dtype == '2I':
            # Convert from meters to microns
            pixel_size = pixel_size*1E-6

        for i in range(len(tif.pages)):
          img = tif.pages[i].asarray()

          # Get the signal channel
          if len(img.shape) == 3:
            # channel is 1-indexed, python is 0-indexed
            img = img[:,:, (channel-1)]

          if frame_shape is None:
            frame_shape = img.shape

          file_name = str(frame_i).zfill(4) + ".tif"
          tifffile.TiffWriter(str(raw_path / file_name)).save(img, resolution=(pixel_size, pixel_size, None))
          frame_i += 1
    spinner.write("Found " + str(frame_i-1) + " images")
    spinner.ok("✅")

  raw_files = raw_path.glob("*.tif")
  raw_files = list(filter(lambda x: x.name[:2] != "._", raw_files))
  raw_files = [ str(x) + "\n" for x in raw_files ]
  raw_files.sort(key=lambda x: str(len(x)) + x)
  with open(str(raw_path / "file_list.txt"), "w") as f:
    f.writelines(raw_files)

  print("Segmenting with CellProfiler")
  cmd = [
    'cellprofiler',
    '-p',
    str(pipeline_path),
    '-c',
    '-r',
    '--file-list=' + str(raw_path / "file_list.txt"),
    '-o',
    str(tiff_path)
  ]
  subprocess.call(cmd)

  # CellProfiler saves masks and other images in the same dir
  # Let's move them
  mask_files = tiff_path.glob("*-mask.tiff")
  for mask_file in mask_files:
    new_name = mask_file.name.replace("-mask.tiff", ".tif")
    mask_file.rename((mask_path / new_name))

  # CellProfiler saves things as .tiff
  # We expect everything as .tif ><
  files = tiff_path.glob("*.tiff")
  for file in files:
    new_name = file.name.replace(".tiff", ".tif")
    file.rename((tiff_path / new_name))

  cp_data = pd.read_csv(str(tiff_path / "ShrunkenNuclei.csv"), header=None)
  categories = cp_data.iloc[0]
  headers = cp_data.iloc[1]
  cp_data = cp_data[2:]

  data = {
    'particle_id': [],
    'mask_id': [],
    'frame': [],

    'x': [],
    'x_px': [],
    'y': [],
    'y_px': [],

    'area': [],
    'mean': [],
    'median': [],
    'min': [],
    'max': [],
    'sum': [],

    'cyto_area': [],
    'cyto_mean': [],
    'cyto_median': [],
    'cyto_min': [],
    'cyto_max': [],
    'cyto_sum': []
  }

  cp_data_2_data = pd.DataFrame([
    [ 'particle_id', 'ShrunkenNuclei', 'ObjectNumber' ],
    [ 'frame', 'Image', 'ImageNumber' ],

    [ 'x', 'ShrunkenNuclei', 'AreaShape_Center_X' ],
    [ 'y', 'ShrunkenNuclei', 'AreaShape_Center_Y' ],

    [ 'area', 'ShrunkenNuclei', 'AreaShape_Area' ],
    [ 'mean', 'ShrunkenNuclei', 'Intensity_MeanIntensity_Orig' ],
    [ 'median', 'ShrunkenNuclei', 'Intensity_MedianIntensity_Orig' ],
    [ 'min', 'ShrunkenNuclei', 'Intensity_MinIntensity_Orig' ],
    [ 'max', 'ShrunkenNuclei', 'Intensity_MaxIntensity_Orig' ],
    [ 'sum', 'ShrunkenNuclei', 'Intensity_IntegratedIntensity_Orig' ],

    [ 'cyto_area', 'Cytoplasm', 'AreaShape_Area' ],
    [ 'cyto_mean', 'Cytoplasm', 'Intensity_MeanIntensity_Orig' ],
    [ 'cyto_median', 'Cytoplasm', 'Intensity_MedianIntensity_Orig' ],
    [ 'cyto_min', 'Cytoplasm', 'Intensity_MinIntensity_Orig' ],
    [ 'cyto_max', 'Cytoplasm', 'Intensity_MaxIntensity_Orig' ],
    [ 'cyto_sum', 'Cytoplasm', 'Intensity_IntegratedIntensity_Orig' ]
  ], columns = [ 'data_col', 'cp_data_cat', 'cp_data_header' ])

  for i,cat in enumerate(categories):
    header = headers[i]

    col = cp_data_2_data.loc[( (cp_data_2_data['cp_data_cat'] == cat) & (cp_data_2_data['cp_data_header'] == header) ), 'data_col']
    if col.shape[0] > 0:
      col = col.iloc[0]
      data[col] = cp_data.iloc[:,i]

  data['mask_id'] = list(map(int, data['particle_id']))
  data['particle_id'] = list(map(lambda x, y: str(x) + "." + str(y), data['frame'], data['particle_id']))
  data['x'] = list(map(float, data['x']))
  data['y'] = list(map(float, data['y']))
  data['x_px'] = list(map(int, data['x']))
  data['y_px'] = list(map(int, data['y']))
  data['area'] = list(map(float, data['area']))
  data['mean'] = list(map(float, data['mean']))
  data['median'] = list(map(float, data['median']))
  data['min'] = list(map(float, data['min']))
  data['max'] = list(map(float, data['max']))
  data['sum'] = list(map(float, data['sum']))
  data['cyto_area'] = list(map(float, data['cyto_area']))
  data['cyto_mean'] = list(map(float, data['cyto_mean']))
  data['cyto_median'] = list(map(float, data['cyto_median']))
  data['cyto_min'] = list(map(float, data['cyto_min']))
  data['cyto_max'] = list(map(float, data['cyto_max']))
  data['cyto_sum'] = list(map(float, data['cyto_sum']))

  data = pd.DataFrame(data)
  data = data.astype({
    'particle_id': 'str',
    'mask_id': 'int',
    'frame': 'int',
    'x_px': 'int',
    'y_px': 'int'
  })

  # Convert pixels to um
  if pixel_size is None:
    pixel_size = 1
    data['unit_conversion'] = 'px'
  else:
    data['unit_conversion'] = 'um/px'

  data['area'] = data['area']*(pixel_size**2)
  data['x'] = data['x']*pixel_size
  data['y'] = data['y']*pixel_size
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size

  # Clean up
  (raw_path / "file_list.txt").unlink()
  (tiff_path / "ShrunkenNuclei.csv").unlink()

  return data

def id_track(group):
  group['track_id'] = (group['frame'].diff() > 1).cumsum()

  return group