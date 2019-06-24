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

def make_tracks(data, frame_num, max_distance=50, gap_size=7):
  cols = data.columns.tolist()
  id_map = build_neighbor_map(data, frame_num, max_distance, gap_size)
  if id_map is None:
    return data

  test = id_map[(id_map['search_particle_id'] == '1.51')]    

  tmp = data.merge(id_map, how='left', on=[ 'particle_id' ])
  idx = ( pd.notnull(tmp['search_particle_id']) )

  tmp.loc[idx, 'particle_id'] = tmp.loc[idx, 'search_particle_id']
  
  return tmp[ cols ]

def build_neighbor_map(data, frame_num, max_distance=50, gap_size=7):
  # Build coordinate list
  ref_data = data[( (data['frame'] <= frame_num+gap_size) & (data['frame'] > frame_num) & ( data['min_frame'] == data['frame'] ))]
  
  if ref_data['frame'].count() < 1:
    return None

  x = ref_data['x'].tolist()
  y = ref_data['y'].tolist()

  search_x = data.loc[(data['frame'] == frame_num), 'x'].tolist()
  search_y = data.loc[(data['frame'] == frame_num), 'y'].tolist()

  coords = list(zip(x,y))
  search_coords = list(zip(search_x, search_y))

  tree = spatial.KDTree(coords)

  res = tree.query([ search_coords ], k=gap_size, distance_upper_bound=max_distance*gap_size)
  
  if gap_size > 1:
    distances = res[0][0][...,0:(gap_size)].flatten()
    idxs = res[1][0][...,0:(gap_size)].flatten()
  else:
    distances = res[0][0]
    idxs = res[1][0]

  neighbor_ids = ref_data['particle_id']
  frames = ref_data['frame']

  # Cases where we couldn't find a neighbor get idx of 
  # the number of items we have, and a distance of inf
  neighbor_ids = neighbor_ids.append(pd.Series([ np.nan ]))
  frames = frames.append(pd.Series([ np.nan ]))

  neighbor_ids = pd.Series(neighbor_ids.iloc[idxs]).tolist()
  frames = pd.Series(frames.iloc[idxs]).tolist()
  search_ids = np.repeat(data.loc[(data['frame'] == frame_num), 'particle_id'], gap_size)

  id_map = pd.DataFrame({
    'search_particle_id': search_ids,
    'particle_id': neighbor_ids,
    'distance': distances,
    'ref_frame': frames
  })

  id_map['rel_frame'] = id_map['ref_frame']-frame_num # Time between found frame and current frame
  id_map['max_distance'] = id_map['rel_frame']*max_distance

  # Remove entries that are beyond the max distance
  id_map.loc[(id_map['distance'] > id_map['max_distance']), 'search_particle_id'] = np.nan

  id_map.sort_values(by=[ 'search_particle_id', 'ref_frame', 'distance' ], ascending=[ True, True, False ], inplace=True)

  # Get the earliest, closest matching particle
  id_map.drop_duplicates(subset='search_particle_id', inplace=True)

  # Sometimes the same ref particle will match 2 different search particles
  id_map.drop_duplicates(subset='particle_id', inplace=True)

  id_map = id_map.loc[( pd.notnull(id_map['search_particle_id']) )]

  return id_map

