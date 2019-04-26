import numpy as np
import pandas as pd

STATS_EVENT_MAP = {
  "R": "Rupture",
  "M": "Mitosis",
  "X": "Apoptosis"
}

def get_cell_stats(p_data, skip_filtered):
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
    result['true' + event] = True if event in p_data['true_event'].unique() else False

  return result

def get_summary_table(results, data_set):
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