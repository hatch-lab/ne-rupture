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
  --segment-pipeline=<string>  [default: preprocessors/cellprofiler/segment.cppipe] The CellProfiler pipelline to use
  --features-pipeline=<string>  [default: preprocessors/cellprofiler/extract.cppipe] The CellProfiler pipelline to use
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
import cv2

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

NAME      = "cellprofiler"


def get_schema():
  return {
  'cellprofiler': lambda x: True,
    '--segment-pipeline': Or(None, lambda n: (ROOT_PATH / n).is_file(), error='That pipeline does not exist'),
    '--features-pipeline': Or(None, lambda n: (ROOT_PATH / n).is_file(), error='That pipeline does not exist')
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

def segment(data_path, tiff_path, extracted_path, masks_path, pixel_size=None, channel=1, params={}):
  """
  Given a set of images, segment them and save the masks and processed images

  Arguments:
    data_path Path The path to the raw images
    tiff_path Path The root path where to save processed images and masks
    extracted_path Path The subdirectory in tiff_path to save raw images
    masks_path Path The subdirectory in tiff_path to save masks
    pixel_size float|None How many microns in a mixel
    channel int Which channel to extract
    params dict Other params passed by preprocess

  Return:
    Information about the images
  """

  pipeline_path = Path(params['--segment-pipeline']).resolve() if params['--segment-pipeline'] is not None else ROOT_PATH / ('preprocessors/cellprofiler/segment.cppipeline')
  
  raw_files = extracted_path.glob("*.tif")
  raw_files = list(filter(lambda x: x.name[:2] != "._", raw_files))
  raw_files = [ str(x) + "\n" for x in raw_files ]
  raw_files.sort(key=lambda x: str(len(x)) + x)
  with open(str(extracted_path / "file_list.txt"), "w") as f:
    f.writelines(raw_files)

  print("Segmenting with CellProfiler...")
  cmd = [
    'cellprofiler',
    '-p',
    str(pipeline_path),
    '-c',
    '-r',
    '--file-list=' + str(extracted_path / "file_list.txt"),
    '-o',
    str(tiff_path)
  ]
  subprocess.call(cmd)

  # CellProfiler saves masks and other images in the same dir
  # Let's move them
  mask_files = tiff_path.glob("*-mask.tiff")
  for mask_file in mask_files:
    new_name = mask_file.name.replace("-mask.tiff", ".tif")
    mask_file.rename((masks_path / new_name))

  # CellProfiler saves things as .tiff
  # We expect everything as .tif ><
  files = tiff_path.glob("*.tiff")
  for file in files:
    new_name = file.name.replace(".tiff", ".tif")
    file.rename((tiff_path / new_name))

def extract_features(tiff_path, tracks_path, cyto_tracks_path, pixel_size=1.0, params={}):
  """
  Extract features from segmented masks that have already been assigned to tracks

  Arguments:
    tiff_path Path The path to the folder that holds processed images
    tracks_path Path The subdir in tiff_path that holds track masks
    cyto_tracks_path Path The subdir in tiff_path that holds cyto masks
    pixel_size float The size of pixels
    params dict A dictionary of parameters

  Return:
    pandas.DataFrame The extracted features for each cell
  """
  cyto_tracks_path.mkdir(exist_ok=True, mode=0o755)

  pipeline_path = Path(params['--features-pipeline']).resolve() if params['--features-pipeline'] is not None else ROOT_PATH / ('preprocessors/cellprofiler/extract.cppipeline')
  
  processed_files = tiff_path.glob("*.tif")
  track_masks = tracks_path.glob("*.tif")

  all_files = list(processed_files) + list(track_masks)
  all_files = list(filter(lambda x: x.name[:2] != "._", all_files))
  all_files = [ str(x) + "\n" for x in all_files ]
  all_files.sort(key=lambda x: str(len(x)) + x)

  with open(str(cyto_tracks_path / "file_list.txt"), "w") as f:
    f.writelines(all_files)

  print("Extracting features with CellProfiler...")
  cmd = [
    'cellprofiler',
    '-p',
    str(pipeline_path),
    '-c',
    '-r',
    '--file-list=' + str(cyto_tracks_path / "file_list.txt"),
    '-o',
    str(cyto_tracks_path)
  ]
  subprocess.call(cmd)

  # CellProfiler saves masks as .tiff
  # Let's change to .tif
  mask_files = cyto_tracks_path.glob("*-cyto-mask.tiff")
  for mask_file in mask_files:
    new_name = mask_file.name.replace("-cyto-mask.tiff", ".tif")
    mask_file.rename((cyto_tracks_path / new_name))

  cp_data = pd.read_csv(str(cyto_tracks_path / "ShrunkenNuclei.csv"), header=None)
  categories = cp_data.iloc[0]
  headers = cp_data.iloc[1]
  cp_data = cp_data[2:]

  data = {
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
    [ 'frame', 'Image', 'FileName_Orig' ],

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

  data['frame'] = [ int(x.replace(".tif", "")) for x in data['frame'] ]
  data['x_px'] = [ int(float(x)) if not pd.isna(float(x)) else np.nan for x in data['x'] ]
  data['y_px'] = [ int(float(x)) if not pd.isna(float(x)) else np.nan for x in data['y'] ]
  data['x'] = [ float(x)*pixel_size for x in data['x'] ]
  data['y'] = [ float(x)*pixel_size for x in data['y'] ]
  data['x_conversion'] = [ pixel_size ]*len(data['x'])
  data['y_conversion'] = [ pixel_size ]*len(data['y'])
  data['area'] = [ float(x)*(pixel_size**2) for x in data['area'] ]
  data['mean'] = [ float(x) for x in data['mean'] ]
  data['median'] = [ float(x) for x in data['median'] ]
  data['min'] = [ float(x) for x in data['min'] ]
  data['max'] = [ float(x) for x in data['max'] ]
  data['sum'] = [ float(x) for x in data['sum'] ]
  data['cyto_area'] = [ float(x)*(pixel_size**2) for x in data['cyto_area'] ]
  data['cyto_mean'] = [ float(x) for x in data['cyto_mean'] ]
  data['cyto_median'] = [ float(x) for x in data['cyto_median'] ]
  data['cyto_min'] = [ float(x) for x in data['cyto_min'] ]
  data['cyto_max'] = [ float(x) for x in data['cyto_max'] ]
  data['cyto_sum'] = [ float(x) for x in data['cyto_sum'] ]

  if pixel_size == 1:
    data['unit_conversion'] = [ 'px' ]*len(data['x'])
  else:
    data['unit_conversion'] = [ 'um/px' ]*len(data['x'])

  data = pd.DataFrame(data)
  data.reset_index(inplace=True, drop=True)

  # Y'all, this is the dumbest thing
  # CellProfiler will not give me the object IDs
  # Even if I just ask for the pixel value of the mask
  # it rescales as val/65536. I can't just multiply it
  # back because of float conversions introducing errors
  particle_ids = [0]*data.shape[0]

  for frame_idx in data['frame'].unique():
    frame_file_name = str(frame_idx).zfill(4) + '.tif'
    frame = cv2.imread(str(tracks_path / frame_file_name), cv2.IMREAD_ANYDEPTH)

    f_data = data[(data['frame'] == frame_idx)]
    for row in f_data.itertuples():
      idx = row.Index
      if pd.isna(row.x_px) or pd.isna(row.y_px):
        continue
      x = int(row.x_px)
      y = int(row.y_px)
      particle_ids[idx] = str(int(frame[y][x]))

  data['particle_id'] = particle_ids
  data['mask_id'] = data['particle_id']

  data = data.dropna()
  data = data.astype({
    'particle_id': 'str',
    'mask_id': 'str',
    'frame': 'int',
    'x_px': 'int',
    'y_px': 'int'
  })

  # Clean up
  (cyto_tracks_path / "file_list.txt").unlink()
  (cyto_tracks_path / "ShrunkenNuclei.csv").unlink()

  return data

def id_track(group):
  group['track_id'] = (group['frame'].diff() > 1).cumsum()

  return group