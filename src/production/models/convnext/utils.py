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
  if len(tot_classes) == 0:
    reverse_tot_classes = {0: 'Maltese', 1: 'Leonberger', 2: 'German_shorthaired_pointer', 3: 'Labrador_retriever', 4: 'Border_collie', 5: 'Akita', 6: 'Ibizan_hound', 7: 'English_springer_spaniel', 8: 'Greater_swiss_mountain_dog', 9: 'Irish_setter', 10: 'Bedlington_terrier', 11: 'Giant_schnauzer', 12: 'Norwegian_elkhound', 13: 'Beagle', 14: 'Newfoundland', 15: 'Boxer', 16: 'Glen_of_imaal_terrier', 17: 'Bull_terrier', 18: 'Petit_basset_griffon_vendeen', 19: 'Norwegian_buhund', 20: 'Keeshond', 21: 'Kuvasz', 22: 'Basenji', 23: 'Welsh_springer_spaniel', 24: 'Boykin_spaniel', 25: 'Border_terrier', 26: 'Neapolitan_mastiff', 27: 'Norwegian_lundehund', 28: 'Lhasa_apso', 29: 'Irish_water_spaniel', 30: 'Smooth_fox_terrier', 31: 'Kerry_blue_terrier', 32: 'Pekingese', 33: 'Otterhound', 34: 'American_foxhound', 35: 'German_pinscher', 36: 'Bulldog', 37: 'Bearded_collie', 38: 'Italian_greyhound', 39: 'Papillon', 40: 'English_cocker_spaniel', 41: 'Golden_retriever', 42: 'Anatolian_shepherd_dog', 43: 'Belgian_sheepdog', 44: 'Basset_hound', 45: 'Silky_terrier', 46: 'American_staffordshire_terrier', 47: 'English_setter', 48: 'Parson_russell_terrier', 49: 'Dogue_de_bordeaux', 50: 'American_water_spaniel', 51: 'Chesapeake_bay_retriever', 52: 'Doberman_pinscher', 53: 'German_wirehaired_pointer', 54: 'Black_russian_terrier', 55: 'Norwich_terrier', 56: 'Australian_cattle_dog', 57: 'Nova_scotia_duck_tolling_retriever', 58: 'Chihuahua', 59: 'Afghan_hound', 60: 'Bullmastiff', 61: 'Boston_terrier', 62: 'German_shepherd_dog', 63: 'Mastiff', 64: 'Irish_terrier', 65: 'Dalmatian', 66: 'Portuguese_water_dog', 67: 'Cocker_spaniel', 68: 'Dandie_dinmont_terrier', 69: 'Black_and_tan_coonhound', 70: 'Poodle', 71: 'Greyhound', 72: 'Irish_wolfhound', 73: 'Xoloitzcuintli', 74: 'Plott', 75: 'Field_spaniel', 76: 'Dachshund', 77: 'Pointer', 78: 'Miniature_schnauzer', 79: 'Brussels_griffon', 80: 'Lakeland_terrier', 81: 'Tibetan_mastiff', 82: 'Clumber_spaniel', 83: 'Entlebucher_mountain_dog', 84: 'Briard', 85: 'Old_english_sheepdog', 86: 'Irish_red_and_white_setter', 87: 'Yorkshire_terrier', 88: 'Bernese_mountain_dog', 89: 'Airedale_terrier', 90: 'Chinese_crested', 91: 'Chinese_shar-pei', 92: 'Bloodhound', 93: 'Havanese', 94: 'Bluetick_coonhound', 95: 'Bichon_frise', 96: 'Cane_corso', 97: 'Japanese_chin', 98: 'Wirehaired_pointing_griffon', 99: 'Brittany', 100: 'American_eskimo_dog', 101: 'Gordon_setter', 102: 'Borzoi', 103: 'Great_dane', 104: 'English_toy_spaniel', 105: 'Komondor', 106: 'Great_pyrenees', 107: 'Flat-coated_retriever', 108: 'Cavalier_king_charles_spaniel', 109: 'Canaan_dog', 110: 'Manchester_terrier', 111: 'Saint_bernard', 112: 'Pharaoh_hound', 113: 'Australian_shepherd', 114: 'Cairn_terrier', 115: 'Pomeranian', 116: 'Finnish_spitz', 117: 'Affenpinscher', 118: 'Belgian_malinois', 119: 'Norfolk_terrier', 120: 'French_bulldog', 121: 'Beauceron', 122: 'Chow_chow', 123: 'Collie', 124: 'Belgian_tervuren', 125: 'Curly-coated_retriever', 126: 'Bouvier_des_flandres', 127: 'Cardigan_welsh_corgi', 128: 'Pembroke_welsh_corgi', 129: 'Lowchen', 130: 'Icelandic_sheepdog', 131: 'Australian_terrier', 132: 'Alaskan_malamute'}
  else:
    reverse_tot_classes = {v:k for k, v in tot_classes.items()}

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
