# coding=utf-8

"""Segment images/extract features

Takes raw TIFF images and extracts features

Will generate individual TIFFs.

Usage:
  imaris-preprocessor.py INPUT [--frame-rate=180] [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SD] [--data-set=0] [--pixel-size=0] [--rolling-ball-size=30] [--img-dir=0]


Arguments:
  INPUT Path to TIFF image sequence; include the trailing slash

Options:
  --img-dir=<string> [defaults: INPUT/images] The directory that contains TIFF images of each frame, for outputting videos.
  --filter-window=<float> [defaults: 5.0] The window size used for the median pass filter, in px
  --gamma=<float> [defaults: 0.50] The gamma correction to use
  --channel=<int> [defaults: 2] The channel to keep (ie, the NLS-3xGFP channel)
  --objective=<int> [defaults: 20] The microscope objective (eg, 20 for 20x)
  --microscope=<string> [defaults: SD] "SPE" or "SD"
  --data-set=<string|falsey> [defaults: None] The unique identifier for this data set. If none is supplied, the base file name of each TIFF will be used.
  --pixel-size=<int|0> [defaults: 0] Specifying microscope and objective will automatically determine pixel size. If supplied here, that value will be used instead.
  --rolling-ball-size=<int> [defaults: 30] The rolling ball diameter to use for rolling ball subtraction, in um
  --frame-rate=<int> [defaults: 180] The seconds that elapse between frames
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

import cv2
import pandas as pd

import subprocess
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from time import sleep
import io

import matlab.engine

from feature_extraction import tracks

### Constants
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()
HEIGHT             = 100
WIDTH              = 100

MAKE_VIDEO_PATH    = (ROOT_PATH / ("validate/render-full-video.py")).resolve()
MATLAB_PATH        = (ROOT_PATH / ("feature_extraction/matlab")).resolve()
MATLAB             = matlab.engine.start_matlab()

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / ("input")).resolve()
csv_path = output_path / "data-no-imaris.csv"
raw_path = (input_path / ("images/raw")).resolve()
processed_path = (input_path / (arguments['--img-dir'])) if arguments['--img-dir'] else (input_path / "images").resolve()
data_set = arguments['--data-set'] if arguments['--data-set'] else (input_path).resolve().name
frame_rate = arguments['--frame-rate'] if arguments['--frame-rate'] else 180

filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 2
objective = int(arguments['--objective']) if arguments['--objective'] else 20
microscope = arguments['--microscope'] if arguments['--microscope'] else "SD"
pixel_size = arguments['--pixel-size'] if arguments['--pixel-size'] else 1
rolling_ball_size = int(arguments['--rolling-ball-size']) if arguments['--pixel-size'] else 30

print("Processing TIFFs for mask-generation...")

cmd = [
  str(FIJI_PATH),
  "--headless",
  str(PREPROCESSOR_PATH),
  str(raw_path) + "/",
  str(processed_path) + "/",
  "--filter-window=" + str(filter_window),
  "--gamma=" + str(gamma),
  "--channel=" + str(channel),
  "--objective=" + str(objective),
  "--microscope=" + microscope,
  "--data-set=" + data_set,
  "--pixel-size=" + str(pixel_size),
  "--rolling-ball-size=" + str(rolling_ball_size)
]
# try:
#   subprocess.check_call(cmd)
# except subprocess.CalledProcessError:
#   print(colorize("red", "Unable to process TIFFs"))
#   exit(1)

def process_frames(frame_paths, m_frame_paths, output_path):
  MATLAB.cd(str(MATLAB_PATH))

  out = io.StringIO()
  promise = MATLAB.process_video(frame_paths, m_frame_paths, output_path, background=True, stdout=out)
  while(promise.done() is not True):
    # print("Sup", out.getvalue())
    sleep(0.1)

  return promise.result()

frame_paths = [ str(x) for x in raw_path.glob("*.tif") ]
frame_paths.sort()

# Masking frames
m_frame_paths = [ str(x) for x in (processed_path / data_set).glob("*.tif") ]
m_frame_paths.sort()

print("Extracting features...")
process_frames(frame_paths, m_frame_paths, str(csv_path))
cells = pd.read_csv(str(csv_path), dtype = { 'particle_id': str })

cells['data_set'] = data_set
cells['x_conversion'] = pixel_size
cells['y_conversion'] = pixel_size
cells['event'] = 'N'
cells['frame_rate'] = frame_rate

print("Building tracks...")
for i in tqdm(cells['frame'].unique(), ncols=90, unit="frames"):
  prev_idx = ( 
    (cells['frame'] == (i-1))
  )
  this_idx = ( 
    (cells['frame'] == i)
  )
  prev_frame = cells[prev_idx]
  this_frame = cells[this_idx]

  if len(prev_frame.index) < 1:
    continue

  if len(this_frame.index) < 1:
    break

  id_map = tracks.build_neighbor_map(prev_frame, this_frame, 50)
  cells.loc[(cells['frame'] == i), 'particle_id'] = tracks.track_frame(id_map, cells.loc[( cells['frame'] == i ), :])

# Filter short tracks
# cells = cells.groupby([ 'data_set', 'particle_id' ]).filter(lambda x: x['frame_rate'].iloc[0]*len(x) > 28800)

print("Writing CSV...")
output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
cells.to_csv(str(csv_path), header=True, encoding='utf-8', index=None)
print("  Done")

print("Making movie...")
cmd = [
  "python",
  str(MAKE_VIDEO_PATH),
  str(processed_path),
  str(csv_path),
  data_set,
  str(output_path),
  "--draw-tracks=1"
]
subprocess.call(cmd)