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

  processed_path = (input_path / "images").resolve()

  cmd = [
    str(FIJI_PATH),
    "--headless",
    str(PREPROCESSOR_PATH),
    str(data_path) + "/",
    str(processed_path) + "/",
    "--filter-window=" + str(filter_window),
    "--gamma=" + str(gamma),
    "--channel=" + str(channel),
    "--data-set=" + data_set,
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

  MATLAB.cd(str(MATLAB_PATH))

  out = io.StringIO()

  tmp_path = TEMP_PATH / (str(time()) + ".csv")
  MATLAB.process_video(frame_paths, m_frame_paths, str(tmp_path), stdout=out)
  
  data = pd.read_csv(str(tmp_path), dtype = { 'particle_id': str })
  data['x_conversion'] = pixel_size
  data['y_conversion'] = pixel_size
  data['median'] = data['mean_proc_nuc']
  data['area'] = data['area_nuc']
  data['sum'] = data['mean_proc_nuc']*data['area_nuc']

  print("Building tracks...")
  for i in tqdm(data['frame'].unique(), ncols=90, unit="frames"):
    prev_idx = ( 
      (data['frame'] == (i-1))
    )
    this_idx = ( 
      (data['frame'] == i)
    )
    prev_frame = data[prev_idx]
    this_frame = data[this_idx]

    if len(prev_frame.index) < 1:
      continue

    if len(this_frame.index) < 1:
      break

    id_map = tracks.build_neighbor_map(prev_frame, this_frame, 50)
    data.loc[(data['frame'] == i), 'particle_id'] = tracks.track_frame(id_map, data.loc[( data['frame'] == i ), :])

  data = base_transform(data, params)

  if tmp_path.exists():
    tmp_path.unlink()

  return data


