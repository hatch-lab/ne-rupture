# coding=utf-8

"""Wrapper for Imaris preprocessing step

Takes raw TIFF images and makes them amenable for feature extraction by Imaris.

Will generate individual TIFFs and a TIFF-stack.

Usage:
  imaris-preprocessor.py INPUT OUTPUT [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SD] [--data-set=None] [--pixel-size=None]

Arguments:
  INPUT Path to TIFF image sequence; include the trailing slash
  OUTPUT Path to modified TIFF image sequence; include the trailing slash

Options:
  --filter-window=<float> [defaults: 5.0] The window size used for the median pass filter, in px
  --gamma=<float> [defaults: 0.50] The gamma correction to use
  --channel=<int> [defaults: 2] The channel to keep (ie, the NLS-3xGFP channel)
  --objective=<int> [defaults: 20] The microscope objective (eg, 20 for 20x)
  --microscope=<string> [defaults: SD] "SPE" or "SD"
  --data-set=<string|falsey> [defaults: None] The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<int|0> [defaults: 0] Specifying microscope and objective will automatically determine pixel size. If supplied here, that value will be used instead.
"""
import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/..").resolve()

from docopt import docopt

import subprocess
import glob

### Constant for getting our base input dir
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
STACK_MAKER_PATH   = (ROOT_PATH / ("fiji/stack-maker.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()

### Arguments and inputs
arguments = docopt(__doc__, version='NER 0.1')

input_path = Path(arguments['INPUT']).resolve()
output_path  = Path(arguments['OUTPUT']).resolve()

filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 2
objective = int(arguments['--objective']) if arguments['--objective'] else 20
microscope = arguments['--microscope'] if arguments['--microscope'] else "SD"
folder_name = arguments['--data-set'] if arguments['--data-set'] else None
pixel_size = arguments['--pixel-size'] if arguments['--pixel-size'] else ""

processes = set()

# Make output dir if necessary
if folder_name is None:
  folder_name = re.sub(r"([0-9]+)\.tif$", "", os.path.basename(files[0]))

folder_path = (output_path / folder_name).resolve()
folder_path.mkdir(exist_ok=True)

print("Processing frames: \033[1m" + str(input_path) + "\033[0m -> \033[1m" + str(folder_path) + "\033[0m")

cmd = [
  str(FIJI_PATH),
  "--headless",
  str(PREPROCESSOR_PATH),
  str(input_path) + "/",
  str(output_path) + "/",
  "--filter-window=" + str(filter_window),
  "--gamma=" + str(gamma),
  "--channel=" + str(channel),
  "--objective=" + str(objective),
  "--microscope=" + microscope,
  "--data-set=" + folder_name,
  "--pixel-size=" + pixel_size
]
subprocess.call(cmd)

stack_path = (output_path / (folder_name + ".tif")).resolve()
print("Generating TIFF stack: \033[1m" + str(stack_path) + "\033[0m")

cmd = [
  str(FIJI_PATH),
  "--headless",
  str(STACK_MAKER_PATH),
  str(folder_path) + "/",
  str(stack_path)
]
subprocess.call(cmd)

