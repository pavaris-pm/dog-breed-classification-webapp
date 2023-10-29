# Load model directly
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
from PIL import Image

# this is to test the model published in huggingface hub
extractor = AutoImageProcessor.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")
model = ConvNextV2ForImageClassification.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")

# sample image
image = Image.open('<img_file_for_test>')

inputs = extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits.shape)
# we will get access of a class by id2label and label2id class
print(model.config)