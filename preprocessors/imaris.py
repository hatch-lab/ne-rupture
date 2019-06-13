# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / "preprocessors"))

import numpy as np
import pandas as pd
from errors.UnexpectedEOFException import UnexpectedEOFException
from lib import base_transform
from PIL import Image
from tqdm import tqdm

from common.output import colorize

NAME = "imaris"

def get_default_data_path(input_path):
  return input_path / "input"

def process_data(data_path, params):
  frame_rate = params['frame_rate']
  tiff_path = params['tiff_path']

  """"
  What Imaris CSV files we need and their columns
  """
  data_files = {
    'position': { 
      'file': 'Position.csv', 
      'data_cols': [ 'TrackID', 'Time', 'Position X', 'Position Y' ] 
    },
    'area': { 
      'file': 'Area.csv', 
      'data_cols': [ 'TrackID', 'Time', 'Area' ] 
    },
    'sum': { 
      'file': 'Intensity_Sum_Ch=1.csv', 
      'data_cols': [ 'TrackID', 'Time', 'Intensity Sum' ] 
    },
    'median': { 
      'file': 'Intensity_Median_Ch=1.csv', 
      'data_cols': [ 'TrackID', 'Time', 'Intensity Median' ] 
    },
    'oblate_ellipticity': {
      'file': 'Ellipticity_Oblate.csv',
      'data_cols': [ 'TrackID', 'Time', 'Ellipticity (oblate)' ]
    }
  }

  """
  Constant for mappng the Imaris column names to easier ones to type
  """
  COL_MAP = {
    'TrackID': 'particle_id',
    'Time' : 'frame',
    'Position X' : 'x',
    'Position Y' : 'y',
    'Area': 'area',
    'Intensity Median': 'median',
    'Intensity Sum': 'sum',
    'Ellipticity (oblate)': 'oblate_ellipticity'
  }


  ### Collect our data
  data = None
  for name,csv_info in tqdm(data_files.items(), ncols=90, unit="files"):
    file_path = data_path / csv_info['file']
    
    if not file_path.exists():
      print(colorize("yellow", str(file_path) + " not found; skipping"))
      continue

    with open(file_path, 'rb') as f:
      # Find the actual start of CSV data
      # Imaris outpus really silly headers
      found_header = False
      for line in f:
        line = line.decode('utf-8').rstrip()
        if line.find(",") != -1:
          # Extract header
          headers = line.split(",")
          if(len(list(set(headers) & set(csv_info['data_cols']))) > 0):
            found_header = True
            break

      if not found_header:
        raise UnexpectedEOFException("Reached end of " + csv_info['file'] + " without finding headers.")

      this_df = pd.read_csv(f, header=None, names=headers, dtype={ 'TrackID': str })
      this_df = this_df[csv_info['data_cols']]
      if(data is None):
        data = this_df
      else:
        data = pd.merge(data, this_df, how='left', on=[ 'TrackID', 'Time' ])

  data.rename(columns=COL_MAP, inplace=True)
  data['particle_id'] = data['particle_id'].astype(str)

  ### Direct image processing

  # Get pixel to micron conversion factor
  frame_idx = np.min(data['frame'])
  frame_file_name = str(frame_idx).zfill(4) + '.tif'
  frame_path = (tiff_path / frame_file_name).resolve()
  with Image.open(frame_path) as img:
    resolution = img.info['resolution']
    x_conversion = resolution[0]
    y_conversion = resolution[1]

  data['x_conversion'] = x_conversion
  data['y_conversion'] = y_conversion

  ### Make particle IDs less… long.
  data['particle_id'] = data['particle_id'].str.replace(r"^1[0]+", "")
  data['particle_id'] = data['particle_id'].str.replace(r"^$", "000")

  data = base_transform(data, params)

  return data
