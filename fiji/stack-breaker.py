# coding=utf-8

"""FIJI-based TIFF pre-processor

Uses Jython, FIJI's Python2 interpreter. Note: This is --->Python2<--. The rest of this library uses Python3.

Usage:
  stack-breaker.py INPUT OUTPUT

Arguments:
  INPUT The TIFF stack we want to break apart
  OUTPUT Where the Tiff files should be saved
"""
import sys
import os

from common.docopt import docopt

from ij import IJ
from ij import ImagePlus
from ij import ImageStack
from ij import CompositeImage
from ij.io import FileSaver

import glob

arguments = docopt(__doc__, version='NER 0.1')

input_file = arguments['INPUT']
output_dir = arguments['OUTPUT']

img = IJ.openImage(input_file)

dims = img.getDimensions()
width = dims[0]
height = dims[1]
num_channels = dims[2]
num_frames = dims[4] # If it's the T index
if num_frames == 1:
  num_frames = dims[3] # If it's the Z index
  img.setDimensions(num_channels, 1, num_frames)


for i in range(1,num_frames+1):
  frame = ImageStack(width, height)
  img.setT(i)
  for c in range(1, num_channels+1):
    img.setC(c)
    ip = img.getProcessor()
    frame.addSlice(ip)

  frame_img = ImagePlus("Frame " + str(i), frame)
  comp = CompositeImage(frame_img, CompositeImage.GRAYSCALE)
  FileSaver(comp).saveAsTiff(output_dir + "/" + str(i).zfill(4) + ".tif")
