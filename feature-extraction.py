# coding=utf-8

"""Segment images/extract features

Takes raw TIFF images and extracts features

Will generate individual TIFFs.

Usage:
  imaris-preprocessor.py INPUT [--filter-window=5.0] [--gamma=0.50] [--channel=2] [--objective=20] [--microscope=SD] [--data-set=0] [--pixel-size=0] [--rolling-ball-size=30] [--img-dir=0]


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
from multiprocessing import Pool, cpu_count

import progressbar
from time import sleep

from feature_extraction import watershed

### Constants
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()
HEIGHT             = 100
WIDTH              = 100

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
output_path = (input_path / ("input")).resolve()
csv_path = output_path / "data-no-imaris.csv"
raw_path = (input_path / ("images/raw")).resolve()
processed_path = (input_path / (arguments['--img-dir'])) if arguments['--img-dir'] else (input_path / "images").resolve()
data_set = arguments['--data-set'] if arguments['--data-set'] else (input_path).resolve().name

filter_window = float(arguments['--filter-window']) if arguments['--filter-window'] else 5.0
gamma = float(arguments['--gamma']) if arguments['--gamma'] else 0.50
channel = int(arguments['--channel']) if arguments['--channel'] else 2
objective = int(arguments['--objective']) if arguments['--objective'] else 20
microscope = arguments['--microscope'] if arguments['--microscope'] else "SD"
pixel_size = arguments['--pixel-size'] if arguments['--pixel-size'] else 1
rolling_ball_size = int(arguments['--rolling-ball-size']) if arguments['--pixel-size'] else 30

def apply_parallel(frame_paths, m_frame_paths, fn, *args):
  """
  Function for parallelizing frame processing

  Will take each raw/processed frame and pass 
  that data to the provided function, along with the 
  supplied arguments.

  Arguments:
    frame_paths list List of frame paths
    m_frame_paths list List of processed frame paths
    fn function The function called with a group as a parameter
    args Arguments to pass through to fn

  Returns:
    Pandas DataFrame The re-assembled data.
  """

  with Pool(cpu_count()) as p:
    groups = []
    for i, frame_path in enumerate(frame_paths):
      frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
      m_frame = cv2.imread(str(m_frame_paths[i]), cv2.IMREAD_GRAYSCALE)
      t = tuple([ i, frame, m_frame ]) + tuple(args)
      groups.append(t)
    rs = p.starmap_async(fn, groups)

    widgets=[
      ' [', progressbar.Timer(), '] ',
      progressbar.AnimatedMarker()
    ]

    bar = progressbar.ProgressBar(widgets=widgets)
    bar.start()

    while not rs.ready():
      bar.update()
      sleep(0.1)

  return pd.concat(rs.get(), sort=False)

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

frame_paths = [ x for x in raw_path.glob("*.tif") ]
frame_paths.sort()

# Masking frames
m_frame_paths = [ x for x in (processed_path / data_set).glob("*.tif") ]
m_frame_paths.sort()

print("Extracting features...")
cells = apply_parallel(frame_paths, m_frame_paths, watershed.process_frame)
output_path.mkdir(mode=0o755, parents=True, exist_ok=True)

cells.to_csv(str(csv_path), header=True, encoding='utf-8', index=None)
