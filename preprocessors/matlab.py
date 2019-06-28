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

from tqdm import tqdm
import progressbar
from time import sleep,time
import io

import matlab.engine

from scipy import interpolate
from scipy import spatial
from scipy import io as sio
from statsmodels.tsa.stattools import kpss

from skimage.util import crop, pad

import math

from feature_extraction import tracks
from lib import base_transform

NAME = "matlab"

### Constants
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()
HEIGHT             = 100
WIDTH              = 100

MAKE_VIDEO_PATH    = (ROOT_PATH / ("validate/render-full-video.py")).resolve()
MATLAB_PATH        = (ROOT_PATH / ("matlab")).resolve()
MATLAB             = matlab.engine.start_matlab()
TEMP_PATH          = (ROOT_PATH / "tmp").resolve()


def get_default_data_path(input_path):
  return input_path / "images/raw"

def process_data(data_path, params):
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

  processed_path = (input_path / "images").resolve()

  tiff_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  cmd = [
    str(FIJI_PATH),
    "--headless",
    str(PREPROCESSOR_PATH),
    str(data_path) + "/",
    str(tiff_path) + "/",
    "--filter-window=" + str(filter_window),
    "--gamma=" + str(gamma),
    "--channel=" + str(channel),
    "--pixel-size=" + str(pixel_size),
    "--rolling-ball-size=" + str(rolling_ball_size)
  ]
  try:
    subprocess.check_call(cmd)
  except subprocess.CalledProcessError:
    print(colorize("red", "Unable to process TIFFs"))
    exit(1)

  frame_paths = [ str(x) for x in data_path.glob("*.tif") ]
  frame_paths.sort()

  m_frame_paths = [ str(x) for x in (processed_path / data_set).glob("*.tif") ]
  m_frame_paths.sort()

  TEMP_PATH.mkdir(mode=0o755, parents=True, exist_ok=True)
  tmp_label = str(time())
  tmp_csv_path = TEMP_PATH / (tmp_label + ".csv")
  tmp_mask_path = TEMP_PATH / (tmp_label)

  tmp_mask_path.mkdir(exist_ok=True)

  out = io.StringIO()

  print("Running MATLAB feature extraction...")
  MATLAB.cd(str(MATLAB_PATH))
  promise = MATLAB.process_video(frame_paths, m_frame_paths, pixel_size, str(tmp_csv_path), str(tmp_mask_path), stdout=out, background=True)
  bar = progressbar.ProgressBar(term_width = 35, max_value = progressbar.UnknownLength, widgets=[ progressbar.BouncingBar() ])
  while promise.done() is not True:
    bar.update(1)
    sleep(0.2)

  bar.finish()
  
  data = pd.read_csv(str(tmp_csv_path), dtype = { 'particle_id': str })
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size
  data['median'] = data['mean_proc_nuc']
  data['area'] = data['area_nuc']
  data['sum'] = data['mean_proc_nuc']*data['area_nuc']

  print("Building tracks...")
  frames = sorted(data['frame'].unique())
  data['min_frame'] = (data['particle_id'].str.split(".").str[0]).astype('int')
  data['orig_particle_id'] = data['particle_id']

  for i in tqdm(frames[:-1], ncols=90, unit="frames"):
    data = tracks.make_tracks(data, i, 10, 3)

  data.drop_duplicates(subset=[ 'particle_id', 'frame' ], inplace=True)

  data = base_transform(data, params)

  # Build MIP for each particle
  MATLAB.cd(str(tmp_mask_path))
  particle_imgs = {} # MIP over the entire video
  ref_particle_imgs = {} # MIP for the first 3 frames
  captured_frames = {} # Number of frames we've captured per pid
  print("Building MIP for each particle...")
  for i in tqdm(frames, ncols=90, unit="frames"):
    mask = np.matrix(sio.loadmat(str(tmp_mask_path / (str(i) + ".mat")))['Lnuc'], dtype=np.uint8)
    img = cv2.imread(m_frame_paths[(i-1)], cv2.IMREAD_GRAYSCALE)
    
    for pid in data.loc[(data['frame'] == i), 'particle_id']:
      
      # Convert the mask ID (which uses pre-track-building IDs)
      # to the new ID as built in tracks.make_tracks
      matlab_id = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), 'orig_particle_id']
      matlab_id = int(matlab_id.iloc[0].split(".")[1])

      this_mask = mask.copy()
      this_mask[this_mask != matlab_id] = 0
      this_mask[this_mask == matlab_id] = 1

      if pid not in particle_imgs:
        particle_imgs[pid] = np.zeros((500,500), dtype=np.uint8)
        captured_frames[pid] = 0

      # Get just the masked nucleus
      fg = cv2.bitwise_and(img, img, mask=this_mask)

      # Crop to 500x500, centered on the nuclear mask
      coords = data.loc[( (data['frame'] == i) & (data['particle_id'] == pid) ), [ 'x', 'y' ]]
      x = int(round(coords['x'].iloc[0]/pixel_size))
      y = int(round(coords['y'].iloc[0]/pixel_size))

      fg = hatchvid.crop_frame(fg, x, y, 500, 500)
      
      # Make a MIP of the previous MIP and this img
      particle_imgs[pid] = np.amax([ particle_imgs[pid], fg ], axis=0)
      captured_frames[pid] += 1
      if captured_frames[pid] < 3:
        # Make the reference frame
        ref_particle_imgs[pid] = particle_imgs[pid].copy()

  data['mip_sum'] = 0.0
  data['mip_normalized_sum'] = 0.0
  for pid, img in particle_imgs.items():
    idx = (data['particle_id'] == pid)
    data.loc[idx, 'mip_sum'] = np.sum(img)

    mask = ref_particle_imgs[pid]
    threshold, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

    cyto = cv2.bitwise_and(img, img, mask=mask)

    data.loc[idx, 'mip_sum'] = np.sum(cyto)

  # Clear out the old images
  if mip_path.exists():
    shutil.rmtree(mip_path)
  mip_path.mkdir(mode=0o755, parents=True, exist_ok=True)

  # Write out our MIPs
  for pid, img in particle_imgs.items():
    cv2.imwrite(str(mip_path / (pid + ".tif")), img)
    cv2.imwrite(str(mip_path / (pid + "-ref.tif")), ref_particle_imgs[pid])

  if tmp_csv_path.exists():
    tmp_csv_path.unlink()
  if tmp_mask_path.exists():
    shutil.rmtree(tmp_mask_path)

  return data
