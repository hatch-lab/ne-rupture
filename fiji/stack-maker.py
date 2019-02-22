# coding=utf-8

"""FIJI-based TIFF pre-processor

Uses Jython, FIJI's Python2 interpreter. Note: This is --->Python2<--. The rest of this library uses Python3.

Usage:
  imaris-stack-maker.py INPUT OUTPUT

Arguments:
  INPUT The folder of TIFF files to turn into a Tiff stack; Must include trailing slash
  OUTPUT Where the Tiff stack should be saved
"""
import sys
import os
import re

from common.docopt import docopt

from ij import IJ
from ij import ImagePlus
from ij import ImageStack
from ij.io import FileSaver

import glob

arguments = docopt(__doc__, version='NER 0.1')

input_dir = arguments['INPUT']
output_file = arguments['OUTPUT']

files = glob.glob(input_dir + "*.tif")
files.sort()
first_frame = IJ.openImage(files[0])

img_stack = ImageStack(first_frame.getWidth(), first_frame.getHeight())
for file in files:
  frame = IJ.openImage(file)
  img_stack.addSlice(frame.getProcessor())

FileSaver(ImagePlus("Stack", img_stack)).saveAsTiffStack(output_file)
