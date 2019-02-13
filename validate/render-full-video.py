# coding=utf-8

"""Render video of a cell

Usage:
  render-full-video.py IMAGES_DIR INPUT_CSV DATA_SET OUTPUT_DIR
  render-full-video.py -h | --help
  render-full-video.py --version

Arguments:
  IMAGES_DIR Path to tiffs of full microscopy video
  INPUT_CSV Path to csv file with positions of the cell
  DATA_SET The data set for the specific particle we want to render
  PARTICLE_ID The specific particle_id in data_set to render
  OUTPUT_DIR The folder to output videos to
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from docopt import docopt
import numpy as np
import pandas as pd
import csv
import cv2
import math
import re
from PIL import Image

### Constant
FONT           = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_SCALE     = 1
FONT_COLOR     = (255,255,255)
FONT_LINE_TYPE = 2

# BGR
CIRCLE_COLORS     = {
  "N": (50,50,50),
  "R": (100,100,255),
  "E": (100,255,100),
  "X": (255,255,255),
  "M": (255,100,100)
}
CIRCLE_THICKNESS  = {
  "N": 1,
  "R": 2,
  "E": 2,
  "X": 2,
  "M": 2
}
CIRCLE_RADIUS     = 30
CIRCLE_LINE_TYPE  = cv2.LINE_AA


### Arguments and inputs
arguments = docopt(__doc__, version='NE-classifier 0.1')

images_path = Path(arguments['IMAGES_DIR']).resolve()
csv_path = Path(arguments['INPUT_CSV']).resolve()
data_set = arguments['DATA_SET']
output_path = Path(arguments['OUTPUT_DIR']).resolve()

### Get our data
data = pd.read_csv(str(csv_path), header=0, dtype={ 'particle_id': str })

data = data[[ 'particle_id', 'frame', 'x', 'y', 'event' ]]
data.sort_values('frame')


### Get some frame information
frame_file_name = str(np.min(data['frame'])).zfill(4) + '.tif'
frame_path = (images_path / (data_set + "/" + frame_file_name)).resolve()
raw_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

width = raw_frame.shape[1]
height = raw_frame.shape[0]

# Get our resolution (so we can map x-y coords to pixels)
with Image.open(frame_path) as img:
  resolution = img.info['resolution']
  x_conversion = resolution[0]
  y_conversion = resolution[1]

# Create a blank frame in case we have missing frames
zero_frame = np.zeros(( height, width ))
zero_frame = zero_frame.astype('uint8')


### Initiate our video writer
output_file_path = str((output_path / (data_set + ".mp4")).resolve())

fourcc = cv2.VideoWriter_fourcc(*'avc1')
writer = cv2.VideoWriter(output_file_path, fourcc, 10, (width, height), True)

start_frame_i = np.min(data['frame'])
end_frame_i = np.max(data['frame'])
this_frame_i = start_frame_i


### Loop through and build our movie
while(this_frame_i <= end_frame_i):
  progress = int(30*this_frame_i//end_frame_i)
  bar = "#" * progress + ' ' * (30 - progress)
  print("\r  Building movie for \033[1m" + data_set + "\033[0m |%s|" % bar, end="\r")

  frame_filter = data['frame'] == this_frame_i
  frame_data = data[frame_filter]

  if frame_data['frame'].count() <= 0: # We're missing a frame
    frame = zero_frame
  else:
    frame_file_name = str(this_frame_i).zfill(4) + '.tif'
    frame_path = (images_path / (data_set + "/" + frame_file_name)).resolve()

    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    # Make the frame color
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for index, row in frame_data.iterrows():
      particle_id = row['particle_id']
      x = int(round(row['x']*x_conversion))
      y = int(round(row['y']*y_conversion))
      event = row['event']

      # Add particle_id
      adj_circle_radius = int(round(CIRCLE_RADIUS/x_conversion))
      cv2.circle(frame, (x, y), adj_circle_radius, CIRCLE_COLORS[event], CIRCLE_THICKNESS[event], CIRCLE_LINE_TYPE)
      cv2.putText(frame, particle_id, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_LINE_TYPE)

  writer.write(frame)

  this_frame_i = this_frame_i+1


### Clean up and exit successfully
print()
writer.release()
exit(0)
