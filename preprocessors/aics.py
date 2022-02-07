# coding=utf-8

"""
Gets data into a format readable by classifiers

Usage:
  preprocess.py aics [options] [--] INPUT

Arguments:
  INPUT Path to the directory containing the raw data (CSV files for Imaris, TIFFs for MATLAB)

Options:
  -h, --help
  -v, --version
  --filter-window=<int>  [default: 8] The window size used for the median pass filter, in px
  --gamma=<float>  [default: 0.50] The gamma correction to use
  --rolling-ball-size=<int>  [default: 100] The rolling ball diameter to use for rolling ball subtraction, in um
"""

import sys
import os
from pathlib import Path
import shutil

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

from docopt import docopt
import lib.video as hatchvid

import numpy as np
import pandas as pd
import cv2

import tifffile
from skimage.color import label2rgb
from skimage import measure, exposure, morphology, segmentation, transform
from scipy import ndimage as ndi

from tqdm import tqdm

from external.aicspkg.aicssegmentation.core.pre_processing_utils import intensity_normalization
from external.aicspkg.aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_3d
from external.aicspkg.aicssegmentation.core.MO_threshold import MO
from external.aicspkg.aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

from schema import Schema, And, Or, Use, SchemaError, Optional, Regex

NAME      = "aics"


def get_schema():
  return {
    'aics': lambda x: True,
    Optional('--mip-dir'): And(str, len),
    '--filter-window': And(Use(int), lambda n: n > 0, error='--filter-window must be > 0'),
    '--gamma': And(Use(float), lambda n: n > 0, error='--gamma must be > 0'),
    '--rolling-ball-size': And(Use(int), lambda n: n > 0)
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

  frame_i = 1

  for file_path in tqdm(raw_files, desc="Segmenting images", unit="frames"):
    with tifffile.TiffFile(str(file_path)) as tif:
      img = tif.pages[0].asarray()

      # Pre-processing
      sys.stdout = open(os.devnull, 'w') # Suppress print
      i_norm = intensity_normalization(img, [ 10.0, 5.0 ])
      i_smooth = image_smoothing_gaussian_3d(i_norm, sigma=1)
      sys.stdout = sys.__stdout__

      # Expand image to 3D
      i_smooth = np.repeat(i_smooth[np.newaxis, :, :], 3, axis=0)

      # Masked object thresholding
      pre_seg_1, mo_mask = MO(i_smooth, 'ave', 100, extra_criteria=True, return_object=True)

      # S2 filter for detecting extra spots
      extra_spots = dot_2d_slice_by_slice_wrapper(i_smooth, [[ 2, 0.025 ]])
      pre_seg_2 = np.logical_or(pre_seg_1, extra_spots)

      # S2 filter for detecting dark spots
      dark_spots = dot_2d_slice_by_slice_wrapper(1-i_smooth, [[ 2, 0.025], [1, 0.025]])
      pre_seg_2[dark_spots > 0] = 0

      # Size filtering
      seg = morphology.remove_small_objects(pre_seg_2>0, min_size=400, connectivity=1)

      # Return to 2D
      seg = seg[1, :, :]
      i_smooth = i_smooth[1,:,:]
      i_norm_8bit = exposure.rescale_intensity(i_norm, out_range=( 0, 255 )).astype(np.uint8)
      i_smooth_8bit = exposure.rescale_intensity(i_smooth, out_range=( 0, 255 )).astype(np.uint8)

      # Label regions
      masks = measure.label(seg)
      # Read props
      props = measure.regionprops(masks, intensity_image=i_norm_8bit)

      # Perform a flood fill on all segments
      st_dev = np.std(i_smooth_8bit)
      for region in props:
        centroid = np.round(region.centroid).astype(np.uint32)
        new_mask = segmentation.flood(i_smooth_8bit, ( centroid[0], centroid[1] ), tolerance=st_dev/2)
        new_mask = ndi.morphology.binary_fill_holes(new_mask)
        new_mask = morphology.binary_closing(new_mask,footprint=morphology.disk(4))
        
        # Sanity check the size of our flood
        new_mask = measure.label(new_mask)
        new_props = measure.regionprops(new_mask, intensity_image=i_norm_8bit)
        if len(new_props) <= 0:
          continue
        new_region = new_props[0]
        max_flood_size = (pixel_size**2)*10000 if pixel_size is not None else 10000
        region_size = new_region.area*(pixel_size**2) if pixel_size is not None else new_region.area
        if region_size > max_flood_size:
          # If our flood is bigger than a 10000 um^2
          continue

        # Update masks
        masks[( (new_mask == 1) | (masks == region.label) )] = region.label

      # Write out masks, images
      file_name = str(frame_i).zfill(4) + ".tif"
      tifffile.TiffWriter(str(tiff_path / file_name)).save(i_smooth_8bit, resolution=(pixel_size, pixel_size, None))
      tifffile.TiffWriter(str(masks_path / file_name)).save(masks, resolution=(pixel_size, pixel_size, None))
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
    image = cv2.imread(str(tiff_path / track_path.name), cv2.IMREAD_GRAYSCALE)

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
      cyto_mask[region_mask == 1] = 0

      cyto_tracks[(cyto_mask == 1)] = region.label

      data['particle_id'].append(str(region.label))
      data['mask_id'].append(str(region.label))
      data['frame'].append(frame_i)

      data['x'].append(region.centroid[1])
      data['x_px'].append(int(region.centroid[1]))
      data['y'].append(region.centroid[0])
      data['y_px'].append(int(region.centroid[0]))

      data['area'].append(region.area)
      data['mean'].append(region.mean_intensity)
      data['median'].append(np.median(image[(tracks == region.label)]))
      data['min'].append(region.min_intensity)
      data['max'].append(region.max_intensity)
      data['sum'].append(region.area*region.mean_intensity)

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