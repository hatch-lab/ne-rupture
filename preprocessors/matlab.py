# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

from common.docopt import docopt
from common.output import colorize

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
from statsmodels.tsa.stattools import kpss

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
  print("Processing TIFFs for mask-generation...")

  data_set = params['data_set']
  input_path = params['input_path']
  filter_window = params['filter_window']
  gamma = params['gamma']
  channel = params['channel']
  pixel_size = params['pixel_size']
  rolling_ball_size = params['rolling_ball_size']
  tiff_path = params['tiff_path']

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
  tmp_path = TEMP_PATH / (str(time()) + ".csv")

  out = io.StringIO()

  print("Running MATLAB feature extraction...")
  MATLAB.cd(str(MATLAB_PATH))
  promise = MATLAB.process_video(frame_paths, m_frame_paths, str(tmp_path), stdout=out, background=True)
  bar = progressbar.ProgressBar(term_width = 35, max_value = progressbar.UnknownLength, widgets=[ progressbar.BouncingBar() ])
  while promise.done() is not True:
    bar.update(1)
    sleep(0.2)

  bar.finish()
  
  data = pd.read_csv(str(tmp_path), dtype = { 'particle_id': str })
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size
  data['median'] = data['mean_proc_nuc']
  data['area'] = data['area_nuc']
  data['sum'] = data['mean_proc_nuc']*data['area_nuc']

  print("Building tracks...")
  frames = sorted(data['frame'].unique())
  data['min_frame'] = (data['particle_id'].str.split(".").str[0]).astype('int')
  for i in tqdm(frames[:-1], ncols=90, unit="frames"):
    data = tracks.make_tracks(data, i, 10, 3)

  data.drop_duplicates(subset=[ 'particle_id', 'frame' ], inplace=True)

  data = base_transform(data, params)
  
  if tmp_path.exists():
    tmp_path.unlink()

  return data


