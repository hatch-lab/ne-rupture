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

import cv2
import numpy as np
import pandas as pd

import subprocess
from multiprocessing import Pool, cpu_count

from time import sleep,time
import io
import glob


from tqdm import tqdm
import progressbar

import matlab.engine

from skimage import exposure, img_as_ubyte, filters, morphology
from skimage.util import crop, pad
from skimage.external import tifffile
import scipy.io as sio

from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage import transform as tf

import math

from feature_extraction import tracks
from lib import base_transform

import warnings

NAME = "matlab"

### Constants
HEIGHT             = 100
WIDTH              = 100

MAKE_VIDEO_PATH    = (ROOT_PATH / ("validate/render-full-video.py")).resolve()
MATLAB_PATH        = (ROOT_PATH / ("matlab")).resolve()
MATLAB             = None
TEMP_PATH          = (ROOT_PATH / "tmp").resolve()


def get_default_data_path(input_path):
  """
  Returns the path to raw data
  
  Arguments:
    input_path str The base input path

  Returns
    str The path to raw data
  """
  return input_path / "images/raw"

def process_image(img, channel, filter_window, gamma, pixel_size, rolling_ball_size):
  """
  Process a raw TIFF into something suitable for segmentation by MATLAB
  
  Arguments:
    img numpy.array The image as a 2- or 3-D array
    channel int The channel to extract
    filter_window int The radius of the median filter window (px)
    gamma float The gamma to set the image to
    pixel_size float The pixels / micron for this iamge
    rolling_ball_size int The radius of the disk used for filtering out the background (um). (Not actually rolling ball subtraction; I am faking it with a median filter)

  Return:
    numpy.array The processed, grayscale 8-bit image
  """
  # Get the signal channel
  if len(img.shape) == 3:
    # channel is 1-indexed, python is 0-indexed
    img = img[:,:, (channel-1)]

  # Map signal to entire 16-bit range
  if img.dtype == np.uint16 or img.dtype == "uint16":
    img = exposure.rescale_intensity(img, in_range='image')

  # Convert to 8-bit grayscale
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    img = img_as_ubyte(img)

  filtered_img = np.copy(img)

  # Perform median filtering
  filtered_img = filters.median(filtered_img, selem=np.ones((filter_window, filter_window)), behavior='rank')

  # Perform gamma correction
  filtered_img = exposure.adjust_gamma(filtered_img, gamma)

  # Rolling ball background subtraction
  # cv2_rolling_ball.subtract_background_rolling_ball is ridonculously slow
  # Gonna fake it
  bg = filters.median(filtered_img, selem=morphology.disk(rolling_ball_size*pixel_size))
  filtered_img = filtered_img.astype(np.int16)
  filtered_img = filtered_img-bg
  filtered_img[filtered_img < 0] = 0
  filtered_img = filtered_img.astype(np.uint8)
  filtered_img = exposure.rescale_intensity(filtered_img)
  # img = subtract_background_rolling_ball(img, rolling_ball_size*pixel_size, light_background=False, use_paraboloid=False, do_presmooth=False)

  return ( img, filtered_img )


def process_data(data_path, params):
  """
  Process raw data, segment it, and extract features
  
  Arguments:
    data_path str The path to raw data
    params dict A dictionary of parameters

  Return:
    pandas.DataFrame The extracted features for each cell
  """
  global MATLAB

  print("Processing TIFFs for feature extraction...")

  data_set = params['data_set']
  input_path = params['input_path']
  filter_window = params['filter_window']
  gamma = params['gamma']
  channel = params['channel']
  pixel_size = params['pixel_size']
  rolling_ball_size = params['rolling_ball_size']
  tiff_path = params['tiff_path']
  mip_path = params['mip_path']
  keep_imgs = params['keep_imgs']

  tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)
  
  raw_path = tiff_path / "8-bit"
  raw_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # Get TIFF stacks
  files = glob.glob(str(data_path) + "/*.tif")
  if len(files) <= 0:
    # TODO: Make this...not crappy
    files = glob.glob(str(data_path) + "/*.TIF")

  if len(files) <= 0:
    print("Could not find any TIFF files!")
    exit(1)
  files.sort(key=lambda x: str(len(x)) + x)
  
  # frame_i = 1
  # for file in files:
  #   with tifffile.TiffFile(file) as tif:
  #     if pixel_size is None and 'XResolution' in tif.pages[0].tags:
  #       pixel_size = tif.pages[0].tags['XResolution'].value
  #       dtype = tif.pages[0].tags['XResolution'].dtype

  #       if len(pixel_size) == 2:
  #         pixel_size = pixel_size[0]

  #       if dtype == '1I':
  #         # Convert from inches to microns
  #         pixel_size = pixel_size*3.937E-5
  #       elif dtype == '2I':
  #         # Convert from meters to microns
  #         pixel_size = pixel_size*1E-6

  #     for i in range(len(tif.pages)):
  #       print("  Processing frame " + str(frame_i) + "...")
  #       raw, img = process_image(tif.pages[i].asarray(), channel, filter_window, gamma, pixel_size, rolling_ball_size)

  #       file_name = str(frame_i).zfill(4) + ".tif"
  #       tifffile.TiffWriter(str(tiff_path / file_name)).save(img, resolution=(pixel_size, pixel_size, None))
  #       tifffile.TiffWriter(str(raw_path / file_name)).save(raw, resolution=(pixel_size, pixel_size, None))
  #       frame_i += 1

  frame_paths = list(tiff_path.glob("*.tif"))
  # Filter out ._ files OS X likes to make
  frame_paths = list(filter(lambda x: x.name[:2] != "._", frame_paths))
  raw_paths = [ str(raw_path / x.name) for x in frame_paths]
  frame_paths = [ str(x) for x in frame_paths ]
  frame_paths.sort()
  raw_paths.sort()

  TEMP_PATH.mkdir(mode=0o755, parents=True, exist_ok=True)
  tmp_label = str(time())
  tmp_label = "1576544943.953862"
  tmp_csv_path = TEMP_PATH / (tmp_label + ".csv")
  tmp_mask_path = TEMP_PATH / (tmp_label)

  tmp_mask_path.mkdir(exist_ok=True)

  out = io.StringIO()

  if MATLAB is None:
    print("Starting matlab engine...")
    MATLAB = matlab.engine.start_matlab()

  print("Running MATLAB feature extraction...")
  start = time()
  MATLAB.cd(str(MATLAB_PATH))
  # promise = MATLAB.process_video(frame_paths, pixel_size, str(tmp_csv_path), str(tmp_mask_path), stdout=out, background=True)
  # # MATLAB.process_video(frame_paths, pixel_size, str(tmp_csv_path), str(tmp_mask_path))
  # bar = progressbar.ProgressBar(term_width = 35, max_value = progressbar.UnknownLength, widgets=[ progressbar.BouncingBar() ])
  # while promise.done() is not True:
  #   bar.update(1)
  #   sleep(0.2)

  # bar.finish()
  # print("Finished in " + str((time() - start)/60) + " min")

  data = pd.read_csv(str(tmp_csv_path), dtype = { 'particle_id': str })
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size
  data['median'] = data['mean_nuc']
  data['area'] = data['area_nuc']
  data['sum'] = data['mean_nuc']*data['area_nuc']

  print("Building tracks...")
  frames = sorted(data['frame'].unique())
  data['min_frame'] = (data['particle_id'].str.split(".").str[0]).astype('int')
  data['orig_particle_id'] = data['particle_id']

  for i in tqdm(frames[:-1], ncols=90, unit="frames"):
    data = tracks.make_tracks(data, i, 10, 3)

  data.drop_duplicates(subset=[ 'particle_id', 'frame' ], inplace=True)

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
  for pid in tqdm(pids, ncols=90, unit="particles"):
    writer = cv2.VideoWriter(str(tmp_mask_path / ("test/" + pid + ".mp4")), fourcc, 10, (400, 200), False)
    for i in frames:
      mask = np.matrix(sio.loadmat(str(tmp_mask_path / (str(i) + ".mat")))['Lnuc'], dtype=np.uint8)
      img = cv2.imread(raw_paths[(i-1)], cv2.IMREAD_GRAYSCALE)

      matlab_id = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), 'orig_particle_id']
      if matlab_id.shape[0] <= 0:
        continue
      matlab_id = int(matlab_id.iloc[0].split(".")[1])

      this_mask = mask.copy()
      this_mask[this_mask != matlab_id] = 0
      this_mask[this_mask == matlab_id] = 1

      # Get just the masked nucleus
      this_img = cv2.bitwise_and(img, img, mask=this_mask)

      # Crop to 500x500, centered on the nuclear mask
      coords = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), [ 'x', 'y' ]]
      x = int(round(coords['x'].iloc[0]/pixel_size))
      y = int(round(coords['y'].iloc[0]/pixel_size))

      this_img = hatchvid.crop_frame(this_img, x, y, 200, 200)

      if prev_img is None:
        prev_img = this_img.copy()
        continue

      shift, error, diffphase = register_translation(prev_img, this_img)

      tform = tf.SimilarityTransform(scale=1, rotation=0, translation=shift)
      warped_img = exposure.rescale_intensity(tf.warp(this_img, tform), out_range=(0,np.max(this_img)))

      combined = np.concatenate( ( this_img, warped_img ), axis=1).astype(np.uint8)

      writer.write(combined)

    writer.release()
  exit()

      

  # for i in tqdm(frames, ncols=90, unit="frames"):
  #   mask = np.matrix(sio.loadmat(str(tmp_mask_path / (str(i) + ".mat")))['Lnuc'], dtype=np.uint8)
  #   img = cv2.imread(raw_paths[(i-1)], cv2.IMREAD_GRAYSCALE)
    
  #   for pid in data.loc[(data['frame'] == i), 'particle_id']:
      
  #     # Convert the mask ID (which uses pre-track-building IDs)
  #     # to the new ID as built in tracks.make_tracks
  #     matlab_id = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), 'orig_particle_id']
  #     matlab_id = int(matlab_id.iloc[0].split(".")[1])

  #     this_mask = mask.copy()
  #     this_mask[this_mask != matlab_id] = 0
  #     this_mask[this_mask == matlab_id] = 1

  #     if pid not in particle_imgs:
  #       particle_imgs[pid] = np.zeros((500,500), dtype=np.uint8)
  #       captured_frames[pid] = 0

  #     # Get just the masked nucleus
  #     fg = cv2.bitwise_and(img, img, mask=this_mask)

  #     # Crop to 500x500, centered on the nuclear mask
  #     coords = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), [ 'x', 'y' ]]
  #     x = int(round(coords['x'].iloc[0]/pixel_size))
  #     y = int(round(coords['y'].iloc[0]/pixel_size))

  #     fg = hatchvid.crop_frame(fg, x, y, 500, 500)
      
  #     # Make a MIP of the previous MIP and this img
  #     particle_imgs[pid] = np.amax([ particle_imgs[pid], fg ], axis=0)
  #     captured_frames[pid] += 1
  #     if captured_frames[pid] < 3:
  #       # Make the reference frame
  #       ref_particle_imgs[pid] = particle_imgs[pid].copy()

  #     if keep_imgs:
  #       img_path = (tiff_path / "cells" / pid).resolve()
  #       img_path.mkdir(parents=True, exist_ok=True)

  #       cv2.imwrite(str(img_path / (str(i) + ".tif")), fg)


  # data['mip_sum'] = 0.0
  # data['mip_cyto_sum'] = 0.0
  # data['mip_normalized_sum'] = 0.0
  # for pid, img in particle_imgs.items():
  #   idx = (data['particle_id'] == pid)
  #   data.loc[idx, 'mip_sum'] = np.sum(img)

  #   mask = ref_particle_imgs[pid]
  #   threshold, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

  #   cyto = cv2.bitwise_and(img, img, mask=mask)

  #   data.loc[idx, 'mip_cyto_sum'] = np.sum(cyto)
  #   data.loc[idx, 'mip_normalized_sum'] = np.sum(cyto)/np.sum(img)

  # # Clear out the old images
  # if mip_path.exists():
  #   shutil.rmtree(mip_path)
  # mip_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # # Write out our MIPs
  # for pid, img in particle_imgs.items():
  #   cv2.imwrite(str(mip_path / (pid + ".tif")), img)
  #   cv2.imwrite(str(mip_path / (pid + "-ref.tif")), ref_particle_imgs[pid])

  if tmp_csv_path.exists():
    tmp_csv_path.unlink()
  if tmp_mask_path.exists():
    shutil.rmtree(tmp_mask_path)

  return data
