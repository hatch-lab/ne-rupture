# coding=utf-8

import sys
import os
from pathlib import Path

ROOT_PATH = Path(__file__ + "/../..").resolve()

sys.path.append(str(ROOT_PATH))

from common.docopt import docopt

import numpy as np
import pandas as pd
from scipy import spatial

def build_neighbor_map(prev_frame, this_frame, max_distance, k=1):
  # Build coordinate list
  prev_x = prev_frame['x'].tolist()
  prev_y = prev_frame['y'].tolist()

  this_x = this_frame['x'].tolist()
  this_y = this_frame['y'].tolist()

  prev_coords = list(zip(prev_x,prev_y))
  this_coords = list(zip(this_x,this_y))

  # Build KDTree
  tree = spatial.KDTree(this_coords)

  res = tree.query([ prev_coords ], k=k, distance_upper_bound=max_distance)
  if k > 1:
    distances = res[0][...,(k-1)][0]
    idxs = res[1][...,(k-1)][0]
  else:
    distances = res[0][0]
    idxs = res[1][0]
  
  neighbor_ids = this_frame['particle_id']
  # Cases where we couldn't find a neighbor get idx of 
  # one more than the number of items we have, and a 
  # distance of inf
  neighbor_ids = neighbor_ids.append(pd.Series([ np.nan ]))
  neighbor_ids = pd.Series(neighbor_ids.iloc[idxs]).tolist()

  id_map = pd.DataFrame({
    'prev_particle_id': prev_frame['particle_id'],
    'particle_id': neighbor_ids,
    'distance': distances
  })

  id_map.sort_values(by=[ 'particle_id', 'distance' ], ascending=[ True, False ], inplace=True)
  # Sometimes, multiple previous particles will map to the same current particle
  # We just take the closest
  id_map.drop_duplicates(subset='particle_id', inplace=True)

  return id_map

def track_frame(id_map, frame):
  tmp = frame.merge(id_map, how='left', on=[ 'particle_id' ])
  idx = ( pd.notnull(tmp['prev_particle_id']) )

  tmp.loc[idx, 'particle_id'] = tmp.loc[idx, 'prev_particle_id']

  return tmp['particle_id'].tolist()

