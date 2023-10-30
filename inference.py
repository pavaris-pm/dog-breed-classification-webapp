# Load model directly
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
import argparse
from PIL import Image

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-img_path", "--img_path", required=True,
   help="Path of Sample Image")

args = vars(ap.parse_args())

# this is to test the model published in huggingface hub
extractor = AutoImageProcessor.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")
model = ConvNextV2ForImageClassification.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")
idx2label =  model.config.id2label

# sample image
image = Image.open(args['img_path'])

inputs = extractor(image, return_tensors="pt")

# the model will output as a raw logits
with torch.no_grad():
    logits = model(**inputs).logits

print('prediction shape of [batch_size, num_classes]:', logits.shape)

# this will predict as a idx
prediction = torch.softmax(logits, dim=1).argmax(dim=1)

cls_prediction = idx2label[prediction.item()]
