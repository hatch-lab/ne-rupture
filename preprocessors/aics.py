# coding=utf-8

import sys
import os
from pathlib import Path
import shutil

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

from common.docopt import docopt
from common.output import colorize
import common.video as hatchvid

import numpy as np
import pandas as pd
import glob
from time import time

import cv2

from skimage.external import tifffile
from skimage.color import label2rgb
from skimage import measure, exposure, morphology, segmentation, transform
from scipy import ndimage as ndi

from skimage.feature import register_translation

from aicspkg.aicssegmentation.core.pre_processing_utils import intensity_normalization
from aicspkg.aicssegmentation.core.pre_processing_utils import image_smoothing_gaussian_3d
from aicspkg.aicssegmentation.core.MO_threshold import MO
from aicspkg.aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper

from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

from feature_extraction import tracks
from lib import base_transform

NAME      = "aics"
TEMP_PATH = (ROOT_PATH / "tmp").resolve()

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

  data_set = params['data_set']
  input_path = params['input_path']
  channel = params['channel']
  pixel_size = params['pixel_size']
  tiff_path = params['tiff_path']
  mip_path = params['mip_path']
  keep_imgs = params['keep_imgs']

  tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)
  
  raw_path = tiff_path / "8-bit"
  raw_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # Get TIFF stacks
  files = list(data_path.glob("*.tif"))
  if len(files) <= 0:
    # TODO: Make this...not crappy
    files = list(data_path.glob("*.TIF"))
  # Filter out ._ files OS X likes to make
  files = list(filter(lambda x: x.name[:2] != "._", files))

  if len(files) <= 0:
    print("Could not find any TIFF files!")
    exit(1)
  files = [ str(x) for x in files ]
  files.sort(key=lambda x: str(len(x)) + x)

  TEMP_PATH.mkdir(mode=0o755, parents=True, exist_ok=True)
  tmp_label = str(time())
  tmp_label = "now"
  tmp_mask_path = TEMP_PATH / (tmp_label)

  tmp_mask_path.mkdir(exist_ok=True)

  data = {
    'particle_id': [],
    'mask_id': [],
    'frame': [],
    'x': [],
    'y': [],
    'area': [],
    'mean': [],
    'min': [],
    'max': []
  }

  frame_i = 1
  all_masks = []

  raw_paths = []

  with yaspin(text="Processing TIFFs for feature extraction") as spinner:
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
          spinner.text = "Processing TIFFs for feature extraction (frame " + str(frame_i) + ")"
          img = tif.pages[i].asarray()
          
          # Get the signal channel
          if len(img.shape) == 3:
            # channel is 1-indexed, python is 0-indexed
            img = img[:,:, (channel-1)]

          # Pre-processing
          sys.stdout = open(os.devnull, 'w') # Suppress print
          i_norm = intensity_normalization(img, [ 0.5, 1.5 ])
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
          seg = morphology.remove_small_objects(pre_seg_2>0, min_size=400, connectivity=1, in_place=False)

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
            new_mask = morphology.binary_closing(new_mask,selem=morphology.disk(4))
            
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
            masks[(new_mask == 1)] = region.label

          all_masks.append(masks.copy())

          # Read props
          props = measure.regionprops(masks, intensity_image=i_norm_8bit)

          for region in props:
            data['particle_id'].append(str(frame_i) + "." + str(region.label))
            data['mask_id'].append(region.label)
            data['frame'].append(frame_i)
            data['x'].append(region.centroid[1])
            data['y'].append(region.centroid[0])
            data['area'].append(region.area)
            data['mean'].append(region.mean_intensity)
            data['min'].append(region.min_intensity)
            data['max'].append(region.max_intensity)

          # Write out normalized and smoothed images
          file_name = str(frame_i).zfill(4) + ".tif"
          tifffile.TiffWriter(str(tiff_path / file_name)).save(i_smooth_8bit, resolution=(pixel_size, pixel_size, None))
          tifffile.TiffWriter(str(raw_path / file_name)).save(i_norm_8bit, resolution=(pixel_size, pixel_size, None))
          raw_paths.append(raw_path / file_name)
          frame_i += 1

    spinner.ok("✅")
  all_masks = np.stack(all_masks, axis=0)

  data = pd.DataFrame(data)

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

  # Fake median and sum
  data['median'] = data['mean']
  data['sum'] = data['area']*data['mean']


  # Assign particles to tracks
  print("Building tracks...")
  frames = sorted(data['frame'].unique())
  data['min_frame'] = data['frame']
  data['orig_particle_id'] = data['particle_id']

  for i in tqdm(frames[:-1], ncols=90, unit="frames"):
    data = tracks.make_tracks(data, i, 10, 3)
  data.drop_duplicates(subset=[ 'particle_id', 'frame' ], inplace=True)

  # "Fill in" gaps where we lost tracking
  data.sort_values(by=[ 'particle_id', 'frame' ], inplace=True)
  data['track_id'] = 0
  data = data.groupby([ 'particle_id' ]).apply(id_track)
  missing_particle_ids = data.loc[( data['track_id'] > 0 ), 'particle_id'].unique()

  if len(missing_particle_ids) > 0:
    print("Interpolating cell masks...")
    missing_data = None
    for particle_id in tqdm(missing_particle_ids, ncols=90, unit="cells"):
      p_data = data.loc[(data['particle_id'] == particle_id),:]

      # Find the frame sets that we are missing
      missing_frames = np.setdiff1d(frames, p_data['frame'].unique())
      missing_frames = pd.DataFrame({
        'frame': missing_frames
      })
      missing_frames.sort_values(by=[ 'frame' ])
      missing_frames['track_id'] = (missing_frames['frame'].diff() > 1).cumsum()
      missing_frames.drop_duplicates(subset=[ 'track_id' ], inplace=True)

      min_frame = p_data['min_frame'].iloc[0]
      orig_particle_id = p_data['orig_particle_id'].iloc[0]

      for missing_frame in missing_frames['frame'].unique():
        ref_frame = missing_frame-1
        if ref_frame not in p_data['frame'].unique():
          continue

        # Interpolate data
        missing_data = {
          'particle_id': [],
          'mask_id': [],
          'frame': [],
          'x': [],
          'y': [],
          'area': [],
          'mean': [],
          'min': [],
          'max': [],
          'min_frame': [],
          'orig_particle_id': [],
          'track_id': [],
        }
        
        mask = p_data.loc[( p_data['frame'] == ref_frame ), 'mask_id'].iloc[0]

        # Find ending frame for this gap
        # If there is no end, skip
        stop_frame = p_data.loc[(p_data['frame'] > missing_frame), 'frame'].unique()
        if len(stop_frame) <= 0:
          continue
        stop_frame = np.min(stop_frame)

        if stop_frame == np.max(frames):
          continue

        missing_frame -= 1
        while(missing_frame < stop_frame):
          # Advance
          missing_frame += 1

          ref_frame = missing_frame-1

          file_name = str(missing_frame).zfill(4) + ".tif"
          i_norm_8bit = cv2.imread(str(raw_path / file_name), cv2.IMREAD_GRAYSCALE)
          i_smooth_8bit = cv2.imread(str(tiff_path / file_name), cv2.IMREAD_GRAYSCALE)

          # Frame is 1-indexed, but all_masks is 0-indexed
          masks = all_masks[(ref_frame-1),:,:].copy()
          masks[(masks != mask)] = 0

          props = measure.regionprops(masks, intensity_image=i_norm_8bit)
          if len(props) <= 0:
            continue
          region = props[0]

          st_dev = np.std(i_smooth_8bit)

          centroid = np.round(region.centroid).astype(np.uint32)
          new_mask = segmentation.flood(i_smooth_8bit, ( centroid[0], centroid[1] ), tolerance=st_dev/2)
          new_mask = ndi.morphology.binary_fill_holes(new_mask)
          new_mask = morphology.binary_closing(new_mask,selem=morphology.disk(4))
          new_mask = measure.label(new_mask)

          props = measure.regionprops(new_mask, intensity_image=i_norm_8bit)
          region = props[0]

          max_flood_size = (pixel_size**2)*10000
          if region.area*(pixel_size**2) > max_flood_size:
            # If our flood is bigger than a 10000 um^2
            continue

          # Update masks stack
          masks = all_masks[(missing_frame-1),:,:].copy()
          masks[(new_mask == mask)] = mask
          all_masks[(missing_frame-1),:,:] = masks

          # Add in new data
          missing_data['particle_id'].append(particle_id)
          missing_data['mask_id'].append(mask)
          missing_data['frame'].append(missing_frame)
          missing_data['x'].append(region.centroid[1])
          missing_data['y'].append(region.centroid[0])
          missing_data['area'].append(region.area)
          missing_data['mean'].append(region.mean_intensity)
          missing_data['min'].append(region.min_intensity)
          missing_data['max'].append(region.max_intensity)
          missing_data['min_frame'].append(min_frame)
          missing_data['orig_particle_id'].append(orig_particle_id)
          missing_data['track_id'] = 0

    if missing_data is not None:
      missing_data = pd.DataFrame(missing_data)
      missing_data['unit_conversion'] = data['unit_conversion'].iloc[0]
      missing_data['area'] = missing_data['area']*(pixel_size**2)
      missing_data['x'] = missing_data['x']*pixel_size
      missing_data['y'] = missing_data['y']*pixel_size
      missing_data['x_conversion'] = pixel_size
      missing_data['y_conversion'] = pixel_size
      missing_data['median'] = missing_data['mean']
      missing_data['sum'] = missing_data['area']*missing_data['mean']

      data = pd.concat([ data, missing_data ], sort=False, ignore_index=True)
      data.sort_values(by=[ 'particle_id', 'frame' ], inplace=True)
      data['track_id'] = 0
      data = data.groupby([ 'particle_id' ]).apply(id_track)

  data = base_transform(data, params)

  # Filter out particles that are too near each other
  data = data.loc[(data['nearest_neighbor_distance'] >= 8*pixel_size),:]

  # Build MIP for each particle
  particle_imgs = {} # MIP over the entire video
  ref_particle_imgs = {} # MIP for the first 3 frames
  captured_frames = {} # Number of frames we've captured per pid
  print("Building MIP for each particle...")
  pids = data.loc[:, 'particle_id'].unique()

  prev_img = None

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(str(tmp_mask_path / ("test/test.mp4")), fourcc, 10, (all_masks.shape[1], all_masks.shape[2]), True)
  for i in frames:
    masks = all_masks[(i-1),:,:]
    img = cv2.imread(str(raw_paths[(i-1)]), cv2.IMREAD_GRAYSCALE)
    labelled = label2rgb(masks, image=img)
    labelled = exposure.rescale_intensity(labelled, in_range=(0,1), out_range='uint8').astype(np.uint8)

    writer.write(labelled)
  writer.release()
  exit()

  for pid in tqdm(pids, ncols=90, unit="particles"):
    writer = cv2.VideoWriter(str(tmp_mask_path / ("test/" + pid + ".mp4")), fourcc, 10, (200, 200), False)
    for i in frames:
      mask = all_masks[(i-1),:,:].copy()
      mask_ids = data.loc[( (data['particle_id'] == pid) & (data['frame'] == i) ), 'mask_id'].unique()

      if len(mask_ids) <= 0:
        continue

      mask_id = mask_ids[0]

      mask[( mask != mask_id )] = 0
      mask[( mask !=0 )] = 1
      mask = mask.astype(np.uint8)
      img = cv2.imread(str(raw_paths[(i-1)]), cv2.IMREAD_GRAYSCALE)
      
      # Get just the masked nucleus
      this_img = cv2.bitwise_and(img, img, mask=mask)

      # Crop to 500x500, centered on the nuclear mask
      coords = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), [ 'x', 'y' ]]
      x = int(round(coords['x'].iloc[0]/pixel_size))
      y = int(round(coords['y'].iloc[0]/pixel_size))

      this_img = hatchvid.crop_frame(this_img, x, y, 200, 200)

      if prev_img is None:
        prev_img = this_img.copy()
        continue

      shift, error, diffphase = register_translation(prev_img, this_img)

      tform = transform.SimilarityTransform(scale=1, rotation=0, translation=shift)
      warped_img = exposure.rescale_intensity(transform.warp(this_img, tform), out_range=(0,np.max(this_img)))

      combined = np.concatenate( ( this_img, warped_img ), axis=1).astype(np.uint8)

      writer.write(this_img)

    writer.release()
  exit()

def id_track(group):
  group['track_id'] = (group['frame'].diff() > 1).cumsum()

  return group

