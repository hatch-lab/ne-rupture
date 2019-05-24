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
threshold = None

for i,m_frame_path in enumerate(m_frame_paths):
  frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
  m_frame = cv2.imread(str(m_frame_path), cv2.IMREAD_GRAYSCALE)

  if threshold is None:
    threshold = filters.threshold_yen(m_frame)

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

  label_image = measure.label(labels)
  image_label_overlay = color.label2rgb(label_image, image=m_frame)

  cv2.imshow('open_frame', image_label_overlay)

  # # Find the threshold used to binarize image
  # if threshold is None:
  #   threshold = filters.threshold_yen(m_frame)
  
  # th_frame = cv2.threshold(m_frame, threshold, 255, cv2.THRESH_BINARY)[1]
  # seg_frame = morphology.opening(m_frame, morphology.disk(8))
  # seg_frame = morphology.erosion(seg_frame, morphology.disk(8))
  # seg_frame = morphology.reconstruction(seg_frame, th_frame)

  # distance = ndi.distance_transform_edt(th_frame)
  # local_max = feature.peak_local_max(distance, labels=seg_frame, indices=False, min_distance=30)

  # markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
  # labels = segmentation.watershed(-distance, markers, mask=seg_frame)

  # label_image = measure.label(labels)
  # image_label_overlay = color.label2rgb(label_image, image=m_frame)

  # cv2.imshow('open_frame', seg_frame)


  # th_frame = morphology.closing(m_frame > threshold, morphology.square(3))
  # distance = ndi.distance_transform_edt(th_frame)
  # local_maxi = feature.peak_local_max(
  #   distance, 
  #   indices=False, 
  #   footprint=np.ones((3, 3)), 
  #   labels=th_frame
  # )
  # markers = ndi.label(local_maxi)[0]
  # cleared = segmentation.watershed(-distance, markers, mask=th_frame)

  # label_image = measure.label(cleared)
  # image_label_overlay = color.label2rgb(label_image, image=m_frame)
  # cv2.imshow('image_label_overlay',image_label_overlay)

  # for region in measure.regionprops(label_image):
  #   if region.area >= 100/pixel_size:
  #     print(region.centroid)

  # threshold,th_frame = cv2.threshold(m_frame, 30, 255, cv2.THRESH_BINARY)

  # laplacian = cv2.Laplacian(th_frame,cv2.CV_8U)
  # edges = filters.sobel(th_frame)
  # # threshold,filtered = cv2.threshold(edges, 0.01, 255, cv2.THRESH_BINARY)
  # edges = edges*255
  # edges = edges.astype(np.uint8)
  # # filtered = np.where(edges>0.0001,255,0)
  # # print(np.nonzero(filtered))

  # temp, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

  # nuclei = np.zeros(( th_frame.shape[0], th_frame.shape[1]), np.uint8)
  # nuclei = cv2.drawContours(nuclei, contours, -1, 255, thickness=cv2.FILLED)
  # labels = label(nuclei)
  # particle_props = regionprops(labels)

  # for particle_prop in particle_props:
  #   print(particle_prop.centroidarray)

  # cv2.imshow('m_frame',m_frame)
  # cv2.imshow('edges',edges)
  # cv2.imshow('nuclei',nuclei)

  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows() 
    exit()

cv2.destroyAllWindows()




