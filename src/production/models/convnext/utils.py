import glob
from typing import Tuple, Dict
import streamlit as st
# path checking in file directory
# count data in each fold in each categories

def get_total_class()->Tuple[Dict, Dict]:
  # to return label2idx and idx2label in order to match simplt with number for preds
  c = 0
  tot_classes = {}
  for img_path in glob.glob('/dog-breed-classification-webapp/dog-breeds-classification/dogImages/train/*/*'):
    # to get breed type and remove number out from breed name
    breed_type = img_path.split('/')[-2][4:]
    if breed_type not in tot_classes.keys():
      tot_classes[breed_type] = c
      c += 1

  # convert the label back
  # label2idx, idx2label
  reverse_tot_classes = {v:k for k, v in tot_classes.items()}
  st.text(tot_classes)
  return (tot_classes, reverse_tot_classes)



def visualize_breed_fold(fold: str)->Dict:
  count_stats = {}
  for img_path in glob.glob(f'/workspaces/dog-breed-classification-webapp/dog-breeds-classification/dogImages/{fold}/*/*'):
    # to get breed type and remove number out from breed name
    breed_type = img_path.split('/')[-2][4:]
    if breed_type not in count_stats.keys():
      count_stats[breed_type] = 1
    else:
      count_stats[breed_type] += 1

  return count_stats
