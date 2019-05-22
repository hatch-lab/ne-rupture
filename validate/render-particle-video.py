# coding=utf-8

"""Render video of a cell

Usage:
  render-particle-video.py IMAGES_DIR INPUT_CSV DATA_SET PARTICLE_ID OUTPUT_DIR [--width=<int>] [--height=<int>] [--scale-factor=<float>] [--graph-file=<path>]
  render-particle-video.py -h | --help
  render-particle-video.py --version

Arguments:
  IMAGES_DIR Path to tiffs of full microscopy video
  INPUT_CSV Path to csv file with positions of the cell
  DATA_SET The data set for the specific particle we want to render
  PARTICLE_ID The specific particle_id in data_set to render
  OUTPUT_DIR The folder to output videos to

Options:
  --width=<int> [defaults: 100] 
  --height=<int> [defaults: 100]
  --scale-factor=<float> [defaults: 3.0]
  --graph-file=<str|None> [defaults: None] If supplied, the graph image will be appended to the bottom of the video
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt
from common.version import get_version

import numpy as np
import pandas as pd
import csv
import cv2
import math
import re
import common.video as hatchvid

### Constant
FONT           = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE     = 1
FONT_COLOR     = (255,255,255)
FONT_LINE_TYPE = 1

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

images_path = Path(arguments['IMAGES_DIR']).resolve()
csv_path = Path(arguments['INPUT_CSV']).resolve()
data_set = arguments['DATA_SET']
particle_id = arguments['PARTICLE_ID']
output_path = Path(arguments['OUTPUT_DIR']).resolve()
width = int(arguments['--width']) if arguments['--width'] else 100
height = int(arguments['--height']) if arguments['--height'] else 100
scale_factor = float(arguments['--scale-factor']) if arguments['--scale-factor'] else 3.0
graph_path = Path(arguments['--graph-file']).resolve() if arguments['--graph-file'] else False

def append_graph(frame, graph, this_frame_i, start_frame_i, end_frame_i):
  # Draw a line on the graph
  gc = np.copy(graph)
  x = int(round(gc.shape[1]*(this_frame_i-start_frame_i)/(end_frame_i-start_frame_i)))
  gc = cv2.line(gc, (x, 0), (x, gc.shape[0]), (255, 255, 255))
  frame = np.concatenate((frame, gc), axis=0)

  return frame

### Get data for just our one particle
data = pd.read_csv(csv_path, header=0, dtype={ 'particle_id': str })

p_filter = ( (data['data_set'] == data_set) & (data['particle_id'] == particle_id) )

p_data = data[p_filter]
p_data = p_data[[ 'frame', 'time', 'x', 'y', 'x_conversion', 'y_conversion' ]]
p_data.sort_values('frame')

### Initiate our video writer
# Create a blank frame in case we have missing frames for a particular particle
zero_frame = np.zeros(( height, width ))
zero_frame = cv2.resize(zero_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
zero_frame = zero_frame.astype('uint8')

# Make output path
(output_path / data_set).mkdir(exist_ok=True)
output_file_path = (output_path / (data_set + "/" + particle_id + ".mp4")).resolve()

movie_width = zero_frame.shape[1]
movie_height = zero_frame.shape[0]
if graph_path:
  # Get the graph we'll append to the bottom of the video
  graph = cv2.imread(str(graph_path))
  movie_height += graph.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'avc1')
writer = cv2.VideoWriter(str(output_file_path), fourcc, 10, (movie_width, movie_height), True)

start_frame_i = np.min(p_data['frame'])
end_frame_i = np.max(p_data['frame'])
this_frame_i = start_frame_i

# Get our resolution (so we can map x-y coords to pixels)
frame_file_name = str(this_frame_i).zfill(4) + '.tif'
frame_path = str((images_path / (data_set + "/" + frame_file_name)).resolve())

### Loop through and build our movie
print("  Building movie for \033[1m" + data_set + ":" + particle_id + "\033[0m")
while(this_frame_i <= end_frame_i):
  coords_filter = p_data['frame'] == this_frame_i
  coords = p_data[coords_filter]

  frame_file_name = str(this_frame_i).zfill(4) + '.tif'
  frame_path = (images_path / (data_set + "/" + frame_file_name)).resolve()

  if coords['frame'].count() <= 0 or not frame_path.exists(): # We're missing a frame
    frame = zero_frame
    time_label = ""
  else:
    raw_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

    x = int(round(coords['x'].iloc[0]/coords['x_conversion'].iloc[0]))
    y = int(round(coords['y'].iloc[0]/coords['y_conversion'].iloc[0]))

    # Crop down to just this particle
    frame = hatchvid.crop_frame(raw_frame, x, y, width, height)
    
    # Resize
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    frame = frame.astype('uint8')

    # Make frame text
    hours = math.floor(coords['time'].iloc[0] / 3600)
    minutes = math.floor((coords['time'].iloc[0] - (hours*3600)) / 60)
    seconds = math.floor((coords['time'].iloc[0] - (hours*3600)) % 60)

    time_label = "{:02d}h{:02d}'{:02d}\" ({:d})".format(hours, minutes, seconds, this_frame_i)

  # Make the frame color
  frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
  cv2.putText(frame, time_label, (10, 20), FONT, FONT_SCALE, FONT_COLOR, FONT_LINE_TYPE)

  if graph_path:
    frame = append_graph(frame, graph, this_frame_i, start_frame_i, end_frame_i)

  writer.write(frame)

  this_frame_i = this_frame_i+1

### Clean up and exit successfully
writer.release()
exit(0)

