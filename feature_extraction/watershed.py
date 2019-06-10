# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt

import copy
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Occasional divide-by-zero warnings with regionprops
import pandas as pd
import cv2
from skimage import measure, morphology, color, filters, segmentation, feature, util
from scipy import ndimage as ndi
from scipy import stats
from scipy import spatial

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

def segment_by_julien(m_frame, vse=8, vse2=1, bwao=50):
  Dfb = m_frame.copy()
  # If we generate negative numbers, they'll be converted to 255!
  Dfb[Dfb < 5] = 5 
  Dfb = Dfb - 5

  gradmag = filters.sobel(Dfb)
  gradmag = gradmag / np.max(gradmag)

  # Define markers by image opening and reconstruction
  se = morphology.disk(vse)
  # MatLab constructs disks a little differently; let's match that
  se = se[1:-1,1:-1]
  Jo = morphology.opening(Dfb, se)
  Je = morphology.erosion(Jo, se)
  Jor = morphology.reconstruction(seed=Je, mask=Dfb, method="dilation").astype(np.uint8)

  # Foreground detection
  # This is a little different from MATLAB
  fgm = feature.peak_local_max(Jor, footprint=morphology.disk(30), indices=False, exclude_border=False).astype(np.uint8)
  # fgm = morphology.erosion(fgm, morphology.disk(1))
  # fgm = morphology.remove_small_objects(fgm.astype('bool'), min_size=bwao, connectivity=2).astype(np.uint8)
  
  se2 = morphology.disk(vse2)
  fgm2 = morphology.closing(fgm, se2)
  fgm3 = morphology.erosion(fgm2, se2)
  # If we pass an array of ints, remove_small_objects thinks they are *labels* not *values*
  fgm4 = morphology.remove_small_objects(fgm3.astype('bool'), min_size=bwao, connectivity=2).astype(np.uint8)

  # Local background delimitation
  threshold = filters.threshold_otsu(Jor)
  bw = cv2.threshold(Jor, threshold, 1, cv2.THRESH_BINARY)[1]
  D = ndi.distance_transform_edt(1-bw)
  DL = segmentation.watershed(D, connectivity=2, watershed_line=True)

  bgm = (DL == 0).astype('uint8')

  gradmag2 = imimposemin(gradmag, np.bitwise_or(bgm, fgm4))-0.001

  L = segmentation.watershed(gradmag2, connectivity=2, watershed_line=True)

  rem = stats.mode(L)[0][0][0]
  Ll = L.copy()
  segmentation.clear_border(Ll, in_place=True)
  Ll[L == rem] = 0

  Ll = morphology.dilation(Ll, morphology.disk(2))

  return Ll

def segment_by_lucian(m_frame):
  threshold = filters.threshold_otsu(m_frame)
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

  segmentation.clear_border(labels, in_place=True)

  return labels

def add_cell_props(cells, extra_info, props):
  cells['data_set'].append(extra_info['data_set'])
  cells['particle_id'].append("{}.{}".format(extra_info['frame'], props.label))
  cells['image_type'].append(extra_info['image_type'])
  cells['mask'].append(extra_info['mask'])
  cells['frame'].append(extra_info['frame'])

  # Coordinates are row-column, not x-y
  cells['x'].append(props.centroid[1]*extra_info['pixel_size']) 
  cells['y'].append(props.centroid[0]*extra_info['pixel_size'])
  cells['weighted_x'].append(props.weighted_centroid[1]*extra_info['pixel_size'])
  cells['weighted_y'].append(props.weighted_centroid[0]*extra_info['pixel_size'])
  
  cells['area'].append(props.area*extra_info['pixel_size'])
  cells['mean'].append(props.mean_intensity)
  cells['max'].append(props.max_intensity)
  cells['min'].append(props.min_intensity)
  
  cells['ellipse_eccentricity'].append(props.eccentricity)
  cells['circle_diameter'].append(props.equivalent_diameter)
  cells['ellipse_major_length'].append(props.major_axis_length)
  cells['ellipse_minor_length'].append(props.minor_axis_length)
  
  cells['central_moment_11'].append(props.moments_central[0][0])
  cells['central_moment_12'].append(props.moments_central[0][1])
  cells['central_moment_13'].append(props.moments_central[0][2])
  cells['central_moment_21'].append(props.moments_central[1][0])
  cells['central_moment_22'].append(props.moments_central[1][1])
  cells['central_moment_23'].append(props.moments_central[1][2])
  cells['central_moment_31'].append(props.moments_central[2][0])
  cells['central_moment_32'].append(props.moments_central[2][1])
  cells['central_moment_33'].append(props.moments_central[2][2])

  cells['perimeter'].append(props.perimeter)
  cells['solidity'].append(props.solidity)

  return cells

def process_frame(pixel_size, data_set, frame_num, frame, m_frame):
  cells = {
    'data_set': [],
    'particle_id': [],
    'image_type': [],
    'mask': [],
    'frame': [],
    'x': [],
    'y': [],
    'weighted_x': [],
    'weighted_y': [],
    'area': [],
    'mean': [],
    'max': [],
    'min': [],
    'ellipse_eccentricity': [],
    'circle_diameter': [],
    'ellipse_major_length': [],
    'ellipse_minor_length': [],
    'central_moment_11': [],
    'central_moment_12': [],
    'central_moment_13': [],
    'central_moment_21': [],
    'central_moment_22': [],
    'central_moment_23': [],
    'central_moment_31': [],
    'central_moment_32': [],
    'central_moment_33': [],
    'perimeter': [],
    'solidity': []
  }

  base_info = {
    'data_set': data_set,
    'frame': frame_num,
    'mask': 'nucleus',
    'image_type': 'raw',
    'pixel_size': pixel_size
  }

  # Raw labelling
  # nuc_labels = segment_by_julien(m_frame)
  nuc_labels = segment_by_lucian(m_frame)

  # Fuse nearby regions
  nuc_regions = measure.regionprops(nuc_labels, coordinates='rc')
  # search = {
  #   'id': list(),
  #   'coords': list()
  # }
  # for region in nuc_regions:
  #   search['id'].append(region.label)
  #   search['coords'].append(( region.centroid[1]*pixel_size, region.centroid[0]*pixel_size ))

  # id_map = get_neighbor_map(search, max_distance=50).dropna()
  # id_map['ref_id'] = id_map['ref_id'].astype(int)

  # for index, row in id_map.iterrows():
  #   nuc_labels[nuc_labels == row['search_id']] = row['ref_id']

  # When nuclei are closer than the dilation, one will win
  cyto_labels = morphology.dilation(nuc_labels, morphology.disk(5))
  cyto_labels[nuc_labels > 0] = 0

  nuc_raw_regions = measure.regionprops(nuc_labels, intensity_image=frame, coordinates='rc')
  cyto_raw_regions = measure.regionprops(cyto_labels, intensity_image=frame, coordinates='rc')

  nuc_processed_regions = measure.regionprops(nuc_labels, intensity_image=m_frame, coordinates='rc')
  cyto_processed_regions = measure.regionprops(cyto_labels, intensity_image=m_frame, coordinates='rc')

  for region in nuc_raw_regions:
    info = {
      'data_set': data_set,
      'frame': frame_num,
      'mask': 'nucleus',
      'image_type': 'raw',
      'pixel_size': pixel_size
    }
    cells = add_cell_props(cells, info, region)

  for region in nuc_processed_regions:
    info = {
      'data_set': data_set,
      'frame': frame_num,
      'mask': 'nucleus',
      'image_type': 'processed',
      'pixel_size': pixel_size
    }
    cells = add_cell_props(cells, info, region)

  for region in cyto_raw_regions:
    info = {
      'data_set': data_set,
      'frame': frame_num,
      'mask': 'cytoplasm',
      'image_type': 'raw',
      'pixel_size': pixel_size
    }
    cells = add_cell_props(cells, info, region)
    
  for region in cyto_processed_regions:
    info = {
      'data_set': data_set,
      'frame': frame_num,
      'mask': 'cytoplasm',
      'image_type': 'processed',
      'pixel_size': pixel_size
    }
    cells = add_cell_props(cells, info, region)

  cells = pd.DataFrame(cells)
  cells['particle_id'] = cells['particle_id'].astype(str)

  return cells

def get_neighbor_map(ref_frame, search_frame=None, max_distance=80):
  k = 1
  # Build KDTree
  tree = spatial.KDTree(ref_frame['coords'])

  if search_frame is None:
    search_frame = copy.deepcopy(ref_frame)
    k = 2

  res = tree.query([ search_frame['coords'] ], k=k, distance_upper_bound=max_distance)
  if k == 2:
    distances = res[0][...,1][0]
    idxs = res[1][...,1][0]
  else:
    distances = res[0][0]
    idxs = res[1][0]

  neighbor_ids = ref_frame['id']
  # Cases where we couldn't find a neighbor get idx of 
  # one more than the number of items we have, and a 
  # distance of inf
  neighbor_ids.append(np.nan)
  neighbor_ids = [ neighbor_ids[i] for i in idxs ]

  id_map = pd.DataFrame({
    'search_id': search_frame['id'],
    'ref_id': neighbor_ids,
    'distance': distances
  })

  return id_map