# coding=utf-8

"""
Gets data into a format readable by classifiers

Usage:
  preprocess.py stardist [options] [--] INPUT

Arguments:
  INPUT Path to the directory containing the raw data

Options:
  -h, --help
  -v, --version
  --skip-normalization  [default: False]
  --percentile-low=<float>  [default: 1.0]
  --percentile-high=<float>  [default: 99.0]
  --probability-threshold=<float>  [default: 0.1]
  --overlap-threshold=<float>  [default: 0.4]
  --boundary-exclusion=<float>  [default: 2]
"""

import sys
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage import measure, morphology, exposure, filters

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

from docopt import docopt
from schema import Schema, And, Or, Use, SchemaError, Optional, Regex
import tifffile
from tqdm import tqdm
import cv2
from stardist.models import StarDist2D
from csbdeep.utils import normalize

NAME      = "stardist"

def get_schema():
  return {
    'stardist': lambda x: True,
    '--skip-normalization': Use(bool),
    '--percentile-low': And(Use(float), lambda n: n >= 0.0 and n <= 100.0, error='--percentile-low must be between 0 and 100'),
    '--percentile-high': And(Use(float), lambda n: n >= 0.0 and n <= 100.0, error='--percentile-high must be between 0 and 100'),
    '--probability-threshold': And(Use(float), lambda n: n >= 0.0 and n <= 1.0, error='--probability-threshold must be between 0 and 1'),
    '--overlap-threshold': And(Use(float), lambda n: n >= 0.0 and n <= 1.0, error='--overlap-threshold must be between 0 and 1'),
    '--boundary-exclusion': And(Use(int), lambda n: n > 0, error='--boundary-exlusion must be > 0')
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

def segment(stack, normalized_path, masks_path, pixel_size=None, channel=1, params={}):
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
  model = StarDist2D.from_pretrained('2D_versatile_fluo')

  frame_i = 1
  print("Segmenting images...")
  for corrected_image in stack:
    image = corrected_image if params['--skip-normalization'] else normalize(corrected_image, pmin=params['--percentile-low'], pmax=params['--percentile-high'])
    labels, details = model.predict_instances(
      image, 
      prob_thresh=params['--probability-threshold'],
      nms_thresh=params['--overlap-threshold']
    )

    props = pd.DataFrame(measure.regionprops_table(labels, properties=('label', 'area')))
    remove = np.unique(props['label'].loc[(props['area'] <= 300)])
    labels[(np.isin(labels, remove))] = 0

    # Write out masks, images
    file_name = str(frame_i).zfill(4) + ".tif"
    tifffile.TiffWriter(str(normalized_path / file_name)).save(exposure.rescale_intensity(image, out_range=(0,255)).astype(np.uint8), resolution=(pixel_size, pixel_size, None))
    tifffile.TiffWriter(str(masks_path / file_name)).save(labels, resolution=(pixel_size, pixel_size, None))
    frame_i += 1

def extract_features(stack, tracks_path, cyto_tracks_path, channels, pixel_size=1.0, params={}):
  """
  Extract features from segmented masks that have already been assigned to tracks

  Arguments:
    tiff_path Path The path to the folder that holds processed images
    tracks_path Path The subdir in tiff_path that holds track masks
    cyto_tracks_path Path The subdir in tiff_path that holds cyto masks
    channels list The channels
    pixel_size float The size of pixels
    params dict A dictionary of parameters

  Return:
    pandas.DataFrame The extracted features for each cell
  """

  track_masks = tracks_path.glob("*.tif")
  track_masks = list(filter(lambda x: x.name[:2] != "._", track_masks))

  data = {
    'particle_id': [],
    'mask_id': [],
    'frame': [],

    'x': [],
    'x_px': [],
    'y': [],
    'y_px': [],

    'area': [],
    'cyto_area': []
  }

  for channel in channels:
    data['channel_' + str(channel) + '_mean'] = []
    data['channel_' + str(channel) + '_median'] = []
    data['channel_' + str(channel) + '_min'] = []
    data['channel_' + str(channel) + '_max'] = []
    data['channel_' + str(channel) + '_sum'] = []

    data['channel_' + str(channel) + '_cyto_mean'] = []
    data['channel_' + str(channel) + '_cyto_median'] = []
    data['channel_' + str(channel) + '_cyto_min'] = []
    data['channel_' + str(channel) + '_cyto_max'] = []
    data['channel_' + str(channel) + '_cyto_sum'] = []

  for track_path in tqdm(track_masks):
    tracks = cv2.imread(str(track_path), cv2.IMREAD_UNCHANGED)
    cyto_tracks = cv2.imread(str(cyto_tracks_path / track_path.name), cv2.IMREAD_UNCHANGED)
    
    frame_i = int(track_path.name.replace(".tif", ""))

    # Read props
    props = measure.regionprops(tracks)

    for region in props:
      # Generate cytoplasmic masks
      data['particle_id'].append(str(region.label))
      data['mask_id'].append(str(region.label))
      data['frame'].append(frame_i)

      data['x'].append(region.centroid[1])
      data['x_px'].append(int(region.centroid[1]))
      data['y'].append(region.centroid[0])
      data['y_px'].append(int(region.centroid[0]))

      data['area'].append(region.area)
      data['cyto_area'].append(np.sum((cyto_tracks == region.label)))
      for channel in channels:
        subset = stack[(frame_i-1),:,:,(channel-1)][(tracks == region.label)]
        cyto_subset = stack[(frame_i-1),:,:,(channel-1)][(cyto_tracks == region.label)]

        if len(subset) <= 0:
          data['channel_' + str(channel) + '_mean'].append(np.nan)
          data['channel_' + str(channel) + '_median'].append(np.nan)
          data['channel_' + str(channel) + '_min'].append(np.nan)
          data['channel_' + str(channel) + '_max'].append(np.nan)
          data['channel_' + str(channel) + '_sum'].append(np.nan)
        else:
          data['channel_' + str(channel) + '_mean'].append(np.mean(subset))
          data['channel_' + str(channel) + '_median'].append(np.median(subset))
          data['channel_' + str(channel) + '_min'].append(np.min(subset))
          data['channel_' + str(channel) + '_max'].append(np.max(subset))
          data['channel_' + str(channel) + '_sum'].append(np.sum(subset))

        if len(cyto_subset) <= 0:
          data['channel_' + str(channel) + '_cyto_mean'].append(np.nan)
          data['channel_' + str(channel) + '_cyto_median'].append(np.nan)
          data['channel_' + str(channel) + '_cyto_min'].append(np.nan)
          data['channel_' + str(channel) + '_cyto_max'].append(np.nan)
          data['channel_' + str(channel) + '_cyto_sum'].append(np.nan)
        else:
          data['channel_' + str(channel) + '_cyto_mean'].append(np.mean(cyto_subset))
          data['channel_' + str(channel) + '_cyto_median'].append(np.median(cyto_subset))
          data['channel_' + str(channel) + '_cyto_min'].append(np.min(cyto_subset))
          data['channel_' + str(channel) + '_cyto_max'].append(np.max(cyto_subset))
          data['channel_' + str(channel) + '_cyto_sum'].append(np.sum(cyto_subset))

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
    pixel_size = 1.0
    data['unit_conversion'] = 'px'
  else:
    data['unit_conversion'] = 'um/px'

  data['area'] = data['area']*(pixel_size**2)
  data['cyto_area'] = data['cyto_area']*(pixel_size**2)
  data['x'] = data['x']*pixel_size
  data['y'] = data['y']*pixel_size
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size

  return data


