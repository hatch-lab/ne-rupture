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
from tqdm import tqdm
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
  raw_files = extracted_path.glob("*.tif")
  raw_files = list(filter(lambda x: x.name[:2] != "._", raw_files))
  raw_files = [ str(x) for x in raw_files ]
  raw_files.sort(key=lambda x: str(len(x)) + x)

  model = StarDist2D.from_pretrained('2D_versatile_fluo')

  frame_i = 1
  for file_path in tqdm(raw_files, desc="Segmenting images", unit="frames"):
    with tifffile.TiffFile(str(file_path)) as tif:
      image = tif.pages[0].asarray() if params['--skip-normalization'] else normalize(tif.pages[0].asarray(), pmin=params['--percentile-low'], pmax=params['--percentile-high'])
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
      tifffile.TiffWriter(str(tiff_path / file_name)).save(exposure.rescale_intensity(image, out_range=(0,255)).astype(np.uint8), resolution=(pixel_size, pixel_size, None))
      tifffile.TiffWriter(str(masks_path / file_name)).save(labels, resolution=(pixel_size, pixel_size, None))
      frame_i += 1

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

  for track_path in tqdm(track_masks):
    tracks = cv2.imread(str(track_path), cv2.IMREAD_ANYDEPTH)
    image = cv2.imread(str(tiff_path / ("corrected/" + track_path.name)), cv2.IMREAD_GRAYSCALE)
    print(track_path)

    frame_i = int(track_path.name.replace(".tif", ""))

    # Read props
    props = measure.regionprops(tracks, intensity_image=image)
    cyto_tracks = np.zeros_like(tracks)

    for region in props:
      # Generate cytoplasmic masks
      region_mask = tracks.copy()
      region_mask[(region_mask != region.label)] = 0
      region_mask[(region_mask != 0)] = 1
      region_mask = morphology.binary_dilation(region_mask, morphology.disk(3)).astype(np.uint8)
      cyto_mask = region_mask.copy()
      cyto_mask = morphology.binary_dilation(cyto_mask, morphology.disk(5)).astype(np.uint8)
      cyto_mask[tracks != 0] = 0

      cyto_tracks[(cyto_mask == 1)] = region.label

      data['particle_id'].append(str(region.label))
      data['mask_id'].append(str(region.label))
      data['frame'].append(frame_i)

      data['x'].append(region.centroid[1])
      data['x_px'].append(int(region.centroid[1]))
      data['y'].append(region.centroid[0])
      data['y_px'].append(int(region.centroid[0]))

      data['area'].append(region.area)
      data['mean'].append(region.intensity_mean)
      data['median'].append(np.median(image[(tracks == region.label)]))
      data['min'].append(region.intensity_min)
      data['max'].append(region.intensity_max)
      data['sum'].append(region.area*region.intensity_mean)

      masked_region = np.ma.masked_array(image, mask=np.invert(cyto_mask.astype(bool)))

      data['cyto_area'].append(np.sum(cyto_mask))
      data['cyto_mean'].append(np.ma.mean(masked_region))
      data['cyto_median'].append(np.ma.median(masked_region))
      data['cyto_min'].append(np.ma.min(masked_region))
      data['cyto_max'].append(np.ma.max(masked_region))
      data['cyto_sum'].append(np.ma.sum(masked_region))

    tifffile.TiffWriter(str(cyto_tracks_path / track_path.name)).save(cyto_tracks, resolution=(pixel_size, pixel_size, None))

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


