import numpy as np
import pandas as pd

STATS_EVENT_MAP = {
  "R": "Rupture",
  "M": "Mitosis",
  "X": "Apoptosis"
}

def get_cell_stats(p_data, skip_filtered=False):
  """
  Returns stats on a cell-by-cell basis

  Returns if a cell has ever undergone a given event. Will ignore frames
  that have a filtered column set to True, if skip_filtered is given

  Arguments:
    p_data Panda DataFrame The particle data
    skip_filtered bool Whether to ignore filtered records

  Returns:
    Panda DataFrame Cell-by-cell data frame
  """
  data_set = p_data['data_set'].iloc[0]
  particle_id = p_data['particle_id'].iloc[0]

  result = pd.DataFrame({
    'data_set': [ data_set ],
    'particle_id': [ particle_id ]
  })
  
  if skip_filtered and 'filtered' in p_data.columns:
    p_data = p_data.loc[(p_data['filtered'] == 0)]

  for event,name in STATS_EVENT_MAP.items():
    result['pred' + event] = True if event in p_data['event'].unique() else False
    if 'true_event' not in p_data.columns:
      result['true' + event] = np.nan
    else:
      result['true' + event] = True if event in p_data['true_event'].unique() else False

    if 'cell_event' in p_data.columns:
      result['pred' + event] = True if event in p_data['cell_event'].unique() else False
      if 'true_cell_event' not in p_data.columns:
        result['true' + event] = np.nan
      else:
        result['true' + event] = True if event in p_data['true_cell_event'].unique() else False

  return result

def get_summary_table(results, data_set):
  """
  Returns a performance summary

  Using cell-by-cell data, will give a number of metrics that can
  be used for evaluating performace.

  Arguments:
    results Panda DataFrame The cell-by-cell data frame from get_cell_stats
    data_set string The data set name to associate with this data

  Returns:
    Panda DataFrame Performance data frame
  """
  names = []
  nums_corr_positive = []
  nums_pred_positive = []
  nums_true_positive = []
  nums_corr_negative = []
  nums_pred_negative = []
  nums_true_negative = []
  
  for event,name in STATS_EVENT_MAP.items():
    names.append(name)
    nums_corr_positive.append(results[((results['pred' + event] == True) & (results['true' + event] == True))].shape[0])
    nums_pred_positive.append(results[((results['pred' + event] == True))].shape[0])
    nums_true_positive.append(results[((results['true' + event] == True))].shape[0])
    nums_corr_negative.append(results[((results['pred' + event] == False) & (results['true' + event] == False))].shape[0])
    nums_pred_negative.append(results[((results['pred' + event] == False))].shape[0])
    nums_true_negative.append(results[((results['true' + event] == False))].shape[0])
  
  summary = pd.DataFrame({
    'data_set': [data_set] * len(names),
    'event': names,
    'num_corr_positive': nums_corr_positive,
    'num_pred_positive': nums_pred_positive,
    'num_true_positive': nums_true_positive,
    'num_corr_negative': nums_corr_negative,
    'num_pred_negative': nums_pred_negative,
    'num_true_negative': nums_true_negative
  })

  return summary