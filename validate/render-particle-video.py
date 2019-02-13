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

from docopt import docopt
import numpy as np
import pandas as pd
import csv
import cv2
import math
import re
from PIL import Image

### Arguments and inputs
arguments = docopt(__doc__, version='NE-classifier 0.1')

images_path = Path(arguments['IMAGES_DIR']).resolve()
csv_path = Path(arguments['INPUT_CSV']).resolve()
data_set = arguments['DATA_SET']
particle_id = arguments['PARTICLE_ID']
output_path = Path(arguments['OUTPUT_DIR']).resolve()
width = int(arguments['--width']) if arguments['--width'] else 100
height = int(arguments['--height']) if arguments['--height'] else 100
scale_factor = float(arguments['--scale-factor']) if arguments['--scale-factor'] else 3.0
graph_path = Path(arguments['--graph-file']).resolve() if arguments['--graph-file'] else False

def crop_frame(frame, x, y, width, height, is_color=False):
  y_radius = int(math.floor(height/2))
  x_radius = int(math.floor(width/2))

  # Check our bounds
  if y-y_radius < 0:
    # We need to add a border to the top
    offset = abs(y-y_radius)
    border_size = ( offset, len(frame[0]), 3 ) if is_color else ( offset, len(frame[0]) )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=0)

    y += offset # What was (0, 0) is now (0, [offset])

  if y+y_radius > frame.shape[0]-1:
    # We need to add a border to the bottom
    offset = abs(frame.shape[0]-1-y_radius)
    border_size = ( offset, len(frame[0]), 3 ) if is_color else ( offset, len(frame[0]) )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((frame, border), axis=0)
    # What was (0, 0) is still (0, 0)

  if x-x_radius < 0:
    # We need to add a border to the left
    offset = abs(x-x_radius)
    border_size = ( len(frame), offset, 3 ) if is_color else ( len(frame), offset )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((border, frame), axis=1)

    x += offset # What was (0, 0) is now ([offset], 0)

  if x+x_radius > frame.shape[1]-1:
    # We need to add a border to the left
    offset = abs(frame.shape[1]-1-x_radius)
    border_size = ( len(frame), offset, 3 ) if is_color else ( len(frame), offset )
    border = np.zeros(border_size, dtype=frame.dtype)
    frame = np.concatenate((frame, border), axis=1)
    # What was (0, 0) is still (0, 0)

  left = x-x_radius + (width-2*x_radius) # To account for rounding errors
  right = x+x_radius
  top = y-y_radius + (width-2*y_radius)
  bottom = y+y_radius
  frame = frame[top:bottom, left:right]

  return frame

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
p_data = p_data[[ 'frame', 'x', 'y' ]]
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
with Image.open(frame_path) as img:
  resolution = img.info['resolution']
  x_conversion = resolution[0]
  y_conversion = resolution[1]

### Loop through and build our movie
print("  Building movie for \033[1m" + data_set + ":" + particle_id + "\033[0m")
while(this_frame_i <= end_frame_i):
  coords_filter = p_data['frame'] == this_frame_i
  coords = p_data[coords_filter]

  if coords['frame'].count() <= 0: # We're missing a frame
    frame = zero_frame
  else:
    frame_file_name = str(this_frame_i).zfill(4) + '.tif'
    frame_path = str((images_path / (data_set + "/" + frame_file_name)).resolve())

    raw_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    x = int(round(coords['x'].iloc[0]*x_conversion))
    y = int(round(coords['y'].iloc[0]*y_conversion))

    # Crop down to just this particle
    frame = crop_frame(raw_frame, x, y, width, height)
    
    # Resize
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    frame = frame.astype('uint8')

  # Make the frame color
  frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

  if graph_path:
    frame = append_graph(frame, graph, this_frame_i, start_frame_i, end_frame_i)

  writer.write(frame)

  this_frame_i = this_frame_i+1

### Clean up and exit successfully
writer.release()
exit(0)

