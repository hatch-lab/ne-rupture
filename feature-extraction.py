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
import numpy as np
import cv2
from skimage import measure, morphology, color, filters, segmentation, feature, util
from scipy import ndimage as ndi
from scipy import stats

### Constants
PREPROCESSOR_PATH  = (ROOT_PATH / ("fiji/frames-preprocessor.py")).resolve()
FIJI_PATH          = Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx").resolve()
HEIGHT             = 100
WIDTH              = 100

### Arguments and inputs
arguments = docopt(__doc__, version=get_version())

input_path = (ROOT_PATH / (arguments['INPUT'])).resolve()
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

# From https://github.com/jordan-g/Calcium-Imaging-Analysis/blob/master/imimposemin.py
def imimposemin(I, BW, conn=None, max_value=255):
  if not I.ndim in (2, 3):
    raise Exception("'I' must be a 2-D or 3D array.")

  if BW.shape != I.shape:
    raise Exception("'I' and 'BW' must have the same shape.")

  if BW.dtype is not bool:
    BW = BW != 0

  # set default connectivity depending on whether the image is 2-D or 3-D
  if conn == None:
    if I.ndim == 3:
      conn = 26
    else:
      conn = 8
  else:
    if conn in (4, 8) and I.ndim == 3:
      raise Exception("'conn' is invalid for a 3-D image.")
    elif conn in (6, 18, 26) and I.ndim == 2:
      raise Exception("'conn' is invalid for a 2-D image.")

  # create structuring element depending on connectivity
  if conn == 4:
    selem = morphology.disk(1)
  elif conn == 8:
    selem = morphology.square(3)
  elif conn == 6:
    selem = morphology.ball(1)
  elif conn == 18:
    selem = morphology.ball(1)
    selem[:, 1, :] = 1
    selem[:, :, 1] = 1
    selem[1] = 1
  elif conn == 26:
    selem = morphology.cube(3)

  fm = I.astype(float)

  try:
    fm[BW]                 = -math.inf
    fm[np.logical_not(BW)] = math.inf
  except:
    fm[BW]                 = -float("inf")
    fm[np.logical_not(BW)] = float("inf")

  if I.dtype == float:
    I_range = np.amax(I) - np.amin(I)

    if I_range == 0:
      h = 0.1
    else:
      h = I_range*0.001
  else:
    h = 1

  fp1 = I + h

  g = np.minimum(fp1, fm)

  # perform reconstruction and get the image complement of the result
  if I.dtype == float:
    J = morphology.reconstruction(1 - fm, 1 - g, selem=selem)
    J = 1 - J
  else:
    J = morphology.reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
    J = 255 - J

  try:
    J[BW] = -math.inf
  except:
    J[BW] = -float("inf")

  return J

def segment_by_julien(m_frame):
  threshold = filters.threshold_yen(m_frame)
  th_frame = m_frame.copy()
  th_frame[(th_frame < threshold)] = 0

  # m_frame[(m_frame < threshold)] = 0
  # th_frame = cv2.threshold(m_frame, threshold, 255, cv2.THRESH_BINARY)[1]

  disk = morphology.disk(8)
  frame_eroded = morphology.erosion(th_frame, disk)
  frame_recon = morphology.reconstruction(seed=frame_eroded, mask=th_frame, method='dilation')
  frame_dilated = morphology.dilation(frame_recon, disk)
  frame_recon2 = morphology.reconstruction(seed=frame_dilated, mask=frame_recon, method='erosion')

  # Foreground detection
  kernel = np.zeros((3, 3))
  kernel[0,1] = 1
  kernel[1,] = 1
  kernel[2,1] = 1
  foreground = feature.peak_local_max(frame_recon2, footprint=kernel, indices=False, exclude_border=False).astype('uint8')
  
  disk = morphology.disk(1)
  foreground = morphology.closing(foreground, disk)
  foreground = morphology.erosion(foreground, disk)
  # If we pass an array of ints, remove_small_objects thinks they are *labels* not *values*
  morphology.remove_small_objects(foreground.astype('bool'), min_size=50, connectivity=2, in_place=True)
  _, labeled_fg = cv2.connectedComponents(foreground.astype(np.uint8))

  # Background detection
  threshold = filters.threshold_yen(frame_recon2)
  bin_frame = cv2.threshold(frame_recon2, threshold, 255, cv2.THRESH_BINARY)[1]
  D = ndi.distance_transform_edt(bin_frame)
  D_labels = segmentation.watershed(D)
  background = (D_labels == 0).astype('uint8')

  sobelx = cv2.Sobel(m_frame, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(m_frame, cv2.CV_64F, 0, 1, ksize=3)
  gradient = np.sqrt(sobelx**2 + sobely**2)
  gradient = 255*(gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))
  gradient = imimposemin(gradient, np.bitwise_or(background, foreground))

  labels = segmentation.watershed(gradient, labeled_fg)

  remove = stats.mode(labels)[0][0][0]
  labels_2 = labels.copy()
  segmentation.clear_border(labels_2, in_place=True)
  labels_2[labels == remove] = 0

  return labels_2

def segment_by_lucian(m_frame, threshold):
  m_frame[(m_frame < threshold)] = 0
  th_frame = cv2.threshold(m_frame, threshold, 255, cv2.THRESH_BINARY)[1]

  # m_frame_o = morphology.closing(m_frame, morphology.disk(8))
  m_frame_o = morphology.opening(th_frame, morphology.disk(8))
  # m_frame_o = morphology.erosion(m_frame_o, morphology.disk(8))

  # m_frame_c = morphology.closing(m_frame_o, morphology.disk(8))
  # m_frame_c = morphology.dilation(m_frame_c, morphology.disk(8))
  # m_frame_c = morphology.reconstruction(util.invert(m_frame_c), util.invert(m_frame_o))

  markers = filters.rank.gradient(m_frame_o, morphology.disk(10)) < 10
  markers = ndi.label(markers)[0]
  gradient = filters.rank.gradient(m_frame, morphology.disk(2))
  labels = segmentation.watershed(gradient, markers)

  return labels

threshold = None
for i,m_frame_path in enumerate(m_frame_paths):
  frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
  m_frame = cv2.imread(str(m_frame_path), cv2.IMREAD_GRAYSCALE)

  if threshold is None:
    threshold = filters.threshold_yen(m_frame)

  j_labels = segment_by_julien(m_frame)
  l_labels = segment_by_lucian(m_frame, threshold)
  
  label_image = measure.label(j_labels)
  image_label_overlay_j = color.label2rgb(label_image, image=m_frame)
  cv2.imshow('Julien', image_label_overlay_j)

  label_image = measure.label(l_labels)
  image_label_overlay_l = color.label2rgb(label_image, image=m_frame)
  cv2.imshow('Lucian', image_label_overlay_l)

  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows() 
    exit()

cv2.destroyAllWindows()




