from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
import glob
from .utils import get_total_class
from typing import Tuple
from transformers import AutoFeatureExtractor


# construct a custom dataset to pass it through dataloader
class ConvNextDogDataset(Dataset):
  def __init__(self,
               dir_path: str,
               fold: str,
               extractor: AutoFeatureExtractor, 
               ):
    self.path = list(glob.glob(f'{dir_path}/{fold}/*/*'))
    self.extractor = extractor
    self.label2idx, self.idx2label = get_total_class()

  def load_image(self, index: int) -> Image.Image:
    '''Opens an image via a path and returns it.'''
    image_path = self.path[index]
    return Image.open(image_path)

  def __len__(self) -> int:
    return len(self.path)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
    '''Returns one sample of data, data and label (X, y).'''
    img = self.load_image(index)
    class_name = self.path[index].split('/')[-2][4:]  # to get a class names
    class_idx = self.label2idx[class_name] # get a class index by mapping with the dictionary

    processed_image = self.extractor(img, return_tensors="pt").pixel_values.squeeze(0)

    return processed_image, class_idx # returned untransformed image and label
