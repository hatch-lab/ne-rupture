# coding=utf-8

"""FIJI-based TIFF pre-processor

Uses Jython, FIJI's Python2 interpreter. Note: This is --->Python2<--. The rest of this library uses Python3.

Usage:
  frames-preprocessor.py INPUT OUTPUT [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SPE] [--data-set=0] [--pixel-size=0] [--rolling-ball-size=30]

Arguments:
  INPUT Path to the TIFF files to process; include the trailing slash
  OUTPUT Path to modified TIFF image sequence; include the trailing slash

Options:
  --filter-window=<float> [defaults: 5.0] The window size used for the median pass filter, in px
  --gamma=<float> [defaults: 0.50] The gamma correction to use
  --channel=<int> [defaults: 2] The channel to keep (ie, the NLS-3xGFP channel)
  --objective=<int> [defaults: 20] The microscope objective (eg, 20 for 20x)
  --microscope=<string> [defaults: SPE] "SPE" or "SD"
  --data-set=<string|falsey> [defaults: None] The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<int|0> [defaults: 0] Specifying microscope and objective will automatically determine pixel size. If supplied here, that value will be used instead.
  --rolling-ball-size=<int> [defaults: 30] The rolling ball diameter to use for rolling ball subtraction, in um
"""
import sys
import os
import re

from common.docopt import docopt

from ij import IJ
from ij import ImagePlus
from ij.process import ImageStatistics
from ij.process import ImageConverter
from ij.plugin import ChannelSplitter
from ij.plugin import ContrastEnhancer
from ij.plugin.filter import RankFilters
from ij.plugin.filter import BackgroundSubtracter
from ij.measure import Calibration
from ij.io import FileSaver

import glob

arguments = docopt(__doc__, version='NER 0.1')

input_dir = arguments['INPUT']
output_dir = arguments['OUTPUT']
filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 2
objective = int(arguments['--objective']) if arguments['--objective'] else 20
microscope = arguments['--microscope'] if arguments['--microscope'] else "SD"
folder_name = arguments['--data-set'] if arguments['--data-set'] else None
pixel_size = int(arguments['--pixel-size']) if arguments['--pixel-size'] else None
rolling_ball_size = int(arguments['--rolling-ball-size']) if arguments['--pixel-size'] else 30

channel_splitter = ChannelSplitter()
contrast_enhancer = ContrastEnhancer()
contrast_enhancer.setNormalize(True)

if pixel_size is None:
  if objective == 20 and microscope == "SD":
    pixel_size = 0.5089
  else:
    pixel_size = 1.0

files = glob.glob(input_dir + "*.tif")
files.sort()

frame_i = 1
for file in files:
  print("Processing frame \033[1m" + str(frame_i) + "\033[0m")
  frame = IJ.openImage(file) # ImagePlus

  # Get just the signal channel
  frame_stack = channel_splitter.getChannel(frame, channel) # ImageStack
  frame = ImagePlus("Channel " + str(channel), frame_stack)

  # Map signal to entire 16-bit range
  contrast_enhancer.stretchHistogram(frame, 0.01)
  processor = frame.getProcessor()
  if(processor.getBitDepth() == 32):
    processor.setMinAndMax(0, 1.0)
  else:
    processor.setMinAndMax(0, processor.maxValue())

  frame = ImagePlus("Frame " + str(frame_i), processor)

  # Convert to 8-bit, grayscale
  converter = ImageConverter(frame)
  converter.convertToGray8()

  # Perform median filtering
  processor = frame.getProcessor()

  filters = RankFilters()
  filters.setup("median", frame)
  filters.rank(processor, filter_window, filters.MEDIAN)

  # Perform gamma correction
  processor.gamma(gamma)

  frame = ImagePlus("Frame " + str(frame_i), processor)

  # Rolling ball background subtraction
  processor = frame.getProcessor()

  bg_subtractor = BackgroundSubtracter()
  bg_subtractor.setup("", frame)
  bg_subtractor.rollingBallBackground(processor, rolling_ball_size/pixel_size, False, False, False, False, True)

  frame = ImagePlus("Frame " + str(frame_i), processor)

  # Calibrate pixels
  calibration = Calibration()
  calibration.setUnit("pixel")
  calibration.pixelWidth = pixel_size
  calibration.pixelHeight = pixel_size
  calibration.pixelDepth = 1.0

  frame.setCalibration(calibration)

  # Save to output dir
  file_name = output_dir + folder_name + "/" + str(frame_i).zfill(4) + ".tif"
  FileSaver(frame).saveAsTiff(file_name)

  frame_i = frame_i + 1