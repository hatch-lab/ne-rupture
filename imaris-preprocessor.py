# coding=utf-8

"""Wrapper for Imaris preprocessing step

Takes raw TIFF images and makes them amenable for feature extraction by Imaris.

Will generate individual TIFFs and a TIFF-stack.

Usage:
  imaris-preprocessor.py INPUT [--img-dir=0] [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SD] [--data-set=None] [--pixel-size=None] [--rolling-ball-size=30]

Arguments:
  INPUT Path to TIFF image sequence; include the trailing slash

Options:
  --img-dir=<string> [defaults: INPUT/../images] The directory that contains TIFF images of each frame, for outputting videos.
  --filter-window=<float> [defaults: 5.0] The window size used for the median pass filter, in px
  --gamma=<float> [defaults: 0.50] The gamma correction to use
  --channel=<int> [defaults: 2] The channel to keep (ie, the NLS-3xGFP channel)
  --objective=<int> [defaults: 20] The microscope objective (eg, 20 for 20x)
  --microscope=<string> [defaults: SD] "SPE" or "SD"
  --data-set=<string|falsey> [defaults: None] The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<int|0> [defaults: 0] Specifying microscope and objective will automatically determine pixel size. If supplied here, that value will be used instead.
  --rolling-ball-size=<int> [defaults: 30] The rolling ball diameter to use for rolling ball subtraction, in um
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt
from common.version import get_version
from common.output import colorize
import common.video as hatchvid

import subprocess
import glob
import numpy as np
import pandas as pd
import cv2
from scipy import spatial
from statsmodels.tsa.stattools import kpss

### Constant for getting our base input dir
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
STACK_MAKER_PATH   = (ROOT_PATH / ("fiji/stack-maker.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
tiff_path = input_path / (arguments['--img-dir']) if arguments['--img-dir'] else (input_path / "images").resolve()
data_set = arguments['--data-set'] if arguments['--data-set'] else (input_path ).resolve().name

filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 2
objective = int(arguments['--objective']) if arguments['--objective'] else 20
microscope = arguments['--microscope'] if arguments['--microscope'] else "SD"
pixel_size = arguments['--pixel-size'] if arguments['--pixel-size'] else ""

# raw_path = (tiff_path / "raw").resolve()
# output_path = tiff_path / data_set
# output_path.mkdir(exist_ok=True)

# print("Processing frames: \033[1m" + str(raw_path) + "\033[0m -> \033[1m" + str(output_path) + "\033[0m")

cmd = [
  str(FIJI_PATH),
  "--headless",
  str(PREPROCESSOR_PATH),
  str(input_path) + "/",
  str(tiff_path) + "/",
  "--filter-window=" + str(filter_window),
  "--gamma=" + str(gamma),
  "--channel=" + str(channel),
  "--objective=" + str(objective),
  "--microscope=" + microscope,
  "--data-set=" + data_set,
  "--pixel-size=" + pixel_size
]
# subprocess.call(cmd)

raw_path = ROOT_PATH / ("../For Matt 30 second data/Tiff/")
frames = [ x for x in raw_path.glob("*.tif") ]
frames.sort()

data = pd.read_csv(str(input_path / "input/data.csv"), header=0, dtype={ 'particle_id': str })
data = data[((data['particle_id'] == '096') & (data['data_set'] == 'LD-rotation-shB1'))]
data.sort_values('frame')

height = 100
width = 100

i = np.min(data['frame'])
out = {
  'frame': [],
  'out_mean_intensity': [],
  'in_mean_intensity': [],
  'raw_out_mean_intensity': [],
  'raw_in_mean_intensity': [],
  'out_sum_intensity': [],
  'in_sum_intensity': [],
  'raw_out_sum_intensity': [],
  'raw_in_sum_intensity': [],
  'out_med_intensity': [],
  'in_med_intensity': [],
  'raw_out_med_intensity': [],
  'raw_in_med_intensity': [],
  'truth': []
}
last_mask = False

while(i <= np.max(data['frame'])):
  if i not in data['frame'].unique():
    print(i)
    i = i+1
    continue
  row = data[(data['frame'] == i)].iloc[0]
  frame_i = int(row['frame'])
  output_path = tiff_path / row['data_set']
  frame_file_name = str(frame_i).zfill(4) + '.tif'
  frame_path = (output_path / frame_file_name).resolve()
  all_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
  raw = cv2.imread(str(frames[(i-1)]), cv2.IMREAD_GRAYSCALE)

  # if j < 250:
  #   j = j+1
  #   continue

  # hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2HSV)
  x = int(round(row['x']/row['x_conversion']))
  y = int(round(row['y']/row['y_conversion']))

  frame = hatchvid.crop_frame(all_frame, x, y, width, height)
  th_frame = frame.copy()
  th, th_frame = cv2.threshold(th_frame, 30, 255, cv2.THRESH_BINARY)
  # frame = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
  cropped_raw = hatchvid.crop_frame(raw, x, y, width, height)

  edges = cv2.Canny(th_frame, 2, 100, apertureSize=3)
  # frame2, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

  # filtered_contours = None
  # for j,contour in enumerate(contours):
  #   if np.min(contour) > 0 and np.max(contour) < 100:
  #     if filtered_contours is None:
  #       filtered_contours = contour
  #     else:
  #       filtered_contours = np.append(filtered_contours, contour, axis=0)

  mask = edges.copy()
  mask_mask = np.zeros((height+2, width+2), np.uint8)

  cv2.floodFill(mask, mask_mask, (50, 50), 255)

  
  # if len(filtered_contours) <= 0:
  #   mask = last_mask
  # else:
  #   coords = np.squeeze(filtered_contours, axis=1)
  #   tree = spatial.KDTree(coords)
  #   res = tree.query([ coords ], k=2)
  #   idxs = res[1][...,1][0]

  #   filtered_contours = np.expand_dims(coords[idxs], axis=1)
  #   # filtered_contours = np.expand_dims(filtered_contours, axis=1)

  #   epsilon = 0.001*cv2.arcLength(filtered_contours,True)
  #   ellipse = cv2.approxPolyDP(filtered_contours, epsilon, True)

  #   mask = np.zeros(( height, width ), np.uint8)
  #   mask = cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

  #   # temp = np.zeros(( height, width ))
  #   # temp = cv2.drawContours(temp, filtered_contours, -1, 255, thickness=cv2.FILLED)
  #   # temp = cv2.dilate(temp, kernel, iterations=2)
  #   # temp = cv2.erode(temp, kernel, iterations=2)

  #   # temp2, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

  #   # # contours = max(contours, key=lambda x: cv2.arcLength(x, True))
  #   # mask = np.zeros(( height, width ), np.uint8)
  #   # # for contour in filtered_contours:
  #   # epsilon = 0.001*cv2.arcLength(filtered_contours,True)
  #   # ellipse = cv2.approxPolyDP(filtered_contours, epsilon, True)
  #   # # box = cv2.boxPoints(ellipse)
  #   # # box = np.int0(box)
  #   # if np.max(ellipse[0]) > height or np.min(ellipse[0]) < 0 or np.max(ellipse[1]) > width or np.min(ellipse[1]) < 0:
  #   #   mask = last_mask
  #   # else:
  #   #   mask = cv2.drawContours(mask, [ellipse], -1, 255, thickness=1)
  #   #   mask = cv2.dilate(mask, kernel)
  #   #   mask_mask = np.zeros((height+2, width+2), np.uint8)

  #   #   cv2.floodFill(mask, mask_mask, (ellipse[0][0][0]+5, ellipse[0][0][1]+5), 255)
  #   #   # mask = cv2.drawContours(mask,[box],0,255,2)

  # # mask = np.zeros(( height, width ))
  # # mask = mask.astype('uint8')
  kernel = np.ones((5,5),np.uint8)
  kernel2 = np.ones((30,30),np.uint8)
  # big_mask = cv2.dilate(mask, kernel)

  # mask = cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

  # closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  big_mask = cv2.dilate(mask, kernel2)
  mask = cv2.dilate(mask, kernel)

  # big_mask = cv2.resize(mask, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
  # big_mask = big_mask[20:120,20:120]

  all_mask = cv2.bitwise_and(cv2.bitwise_not(mask),big_mask)

  cropped_raw = cropped_raw - np.mean(raw)
  # frame = frame - np.mean(all_frame)
  raw_out_search = cv2.bitwise_and(cropped_raw, cropped_raw, mask=all_mask)
  raw_in_search = cv2.bitwise_and(cropped_raw, cropped_raw, mask=mask)
  out_search = cv2.bitwise_and(frame, frame, mask=all_mask)
  in_search = cv2.bitwise_and(frame, frame, mask=mask)

  cv2.imshow('frame',frame)
  cv2.imshow('edges',edges)
  # cv2.imshow('temp',temp)
  cv2.imshow('mask',mask)
  # cv2.imshow('closed_mask',closed_mask)
  cv2.imshow('big_mask',big_mask)
  cv2.imshow('all_mask',all_mask)

  last_mask = mask

  out['frame'].append(row['frame'])
  out['out_mean_intensity'].append(np.mean(out_search))
  out['in_mean_intensity'].append(np.mean(in_search))
  out['raw_out_mean_intensity'].append(np.mean(raw_out_search))
  out['raw_in_mean_intensity'].append(np.mean(raw_in_search))
  out['out_sum_intensity'].append(np.sum(out_search))
  out['in_sum_intensity'].append(np.sum(in_search))
  out['raw_out_sum_intensity'].append(np.sum(raw_out_search))
  out['raw_in_sum_intensity'].append(np.sum(raw_in_search))
  out['out_med_intensity'].append(np.median(out_search))
  out['in_med_intensity'].append(np.median(in_search))
  out['raw_out_med_intensity'].append(np.median(raw_out_search))
  out['raw_in_med_intensity'].append(np.median(raw_in_search))
  out['truth'].append(row['true_event'])

  print(i)

  print(row['true_event'], np.sum(raw_out_search)/np.sum(raw_in_search))
  # i = i+1
  c = cv2.waitKey(0)
  if c == 2:
    i = i-1
  else:
    i = i+1
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows() 
    exit()

cv2.destroyAllWindows()
out = pd.DataFrame(out)

def sliding_average(data, window, step, frame_rate):
  """
  Calculates the sliding average of a column

  Arguments:
    data list The data to operate on
    window int The size of the window in seconds
    step int How far to advance the window in seconds
    frame_rate int The number of seconds in a frame

  Returns:
    list The sliding window average of equal length to data
  """
  # window and step are given in seconds
  # We need them in frames
  window = int(window/frame_rate)
  step = int(step/frame_rate)
  total = data.size
  spots = np.arange(1, (total-window)+step, step)
  result = [ 0.0 ]*total

  for i in range(0,len(spots)-1):
    result[spots[i]:(spots[i]+window+1)] = [ np.mean(data.iloc[spots[i]:(spots[i]+window+1)]) ] * len(result[spots[i]:(spots[i]+window+1)])

  return result

def make_stationary(group, col, new_col):
  """
  If a [col] is not stationary, apply 1st order difference

  Arguments:
    group Pandas DataFrame of each particle
    col string The column to test

  Returns:
    Modified Pandas DataFrame
  """
  import warnings
  warnings.simplefilter("ignore")

  group = group.sort_values(by=["frame"])
  frame_rate = 30
  smoothed_mean = sliding_average(group[col], 3600, 1800, frame_rate)
  group[new_col] = group[col] - smoothed_mean
  group.loc[(group[new_col] == group[col]), new_col] = np.nan
    
  # Move mean value to 0
  group[new_col] = group[new_col] - np.mean(group[new_col])

  return group

out = make_stationary(out, 'out_mean_intensity', 'stationary_out')
out = make_stationary(out, 'in_mean_intensity', 'stationary_in')
out = make_stationary(out, 'raw_out_mean_intensity', 'stationary_raw_out')
out = make_stationary(out, 'raw_in_mean_intensity', 'stationary_raw_in')

out = make_stationary(out, 'out_sum_intensity', 'stationary_out_sum')
out = make_stationary(out, 'in_sum_intensity', 'stationary_in_sum')
out = make_stationary(out, 'raw_out_sum_intensity', 'stationary_raw_out_sum')
out = make_stationary(out, 'raw_in_sum_intensity', 'stationary_raw_in_sum')

out = make_stationary(out, 'out_med_intensity', 'stationary_out_med')
out = make_stationary(out, 'in_med_intensity', 'stationary_in_med')
out = make_stationary(out, 'raw_out_med_intensity', 'stationary_raw_out_med')
out = make_stationary(out, 'raw_in_med_intensity', 'stationary_raw_in_med')

out.to_csv(str(ROOT_PATH / "test-matt30s-029.csv"), header=True, encoding='utf-8', index=None)

