from transformers import (
    AutoFeatureExtractor, 
    AutoModelForImageClassification,
    AutoImageProcessor,
    ConvNextV2ForImageClassification,
) 
from typing import List, Tuple, Dict
from .convnext.core import ConvNextTransfer
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import streamlit as st


@st.cache_resource
def load_classifier_model(engine: str)->Tuple:
    # load convnext from huggingface hub
    if engine=='convnext_v1':
        extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224-22k")
        convnext = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224-22k")
    elif engine=='convnext_v2':
        # this is to test the model published in huggingface hub
        extractor = AutoImageProcessor.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")
        convnext = ConvNextV2ForImageClassification.from_pretrained("Pavarissy/ConvNextV2-large-DogBreed")

    return (convnext, extractor)

def init_model(engine:str='convnext_v2', weight_path: str=None)->nn.Module:
    # initilialize a model
    if engine=='convnext_v1':
        extractor , convnext = load_classifier_model(engine)

        conv_block = convnext.convnext
        predictor = ConvNextTransfer(conv_block)

        if weight_path:
            # Load the model's state dictionary
            state_dict = torch.load(weight_path)

            # Load the state dictionary to the model
            predictor.load_state_dict(state_dict)
            return predictor, extractor

    elif engine=='convnext_v2':
        # in case that it is model available on huggingface
        extractor , convnext = load_classifier_model(engine)
        return convnext, extractor
    
def classify(
        image: Image.Image,
        classifier: ConvNextV2ForImageClassification,
        extractor: AutoImageProcessor,
    )->Tuple[
        torch.Tensor, 
        torch.Tensor,
        str,
    ]:

    
    _ , idx2label = classifier.config.label2id, classifier.config.id2label

    # feature extraction
    inputs = extractor(image, return_tensors="pt")
    raw_logits = classifier(inputs.pixel_values).logits
    probs = torch.softmax(raw_logits, dim=1)
    # this will predict as a idx
    prediction = torch.softmax(raw_logits, dim=1).argmax(dim=1)

    cls_prediction = idx2label[prediction.item()]

    return (probs, prediction, cls_prediction)


def plot_probs_distribution(
        classifier: ConvNextV2ForImageClassification,
        probs: torch.Tensor,
    )->plt.subplots:
    
    _ , idx2label = classifier.config.label2id, classifier.config.id2label

    # Get the top 5 maximum values and indices along dimension 1
    top_values, top_indices = torch.topk(probs, k=3, dim=1)

    # Convert tensors to numpy arrays for plotting
    top_values_np = top_values.squeeze().detach().numpy()
    top_indices_np = top_indices.squeeze().detach().numpy()

    #top_values_np_sort = np.sort(top_values_np)[::-1]
    #top_labels = [idx2label[index] for index in list(top_indices_np).sort(reverse=True)]
    top_labels = [idx2label[index] for index in list(top_indices_np)]

    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.barh(top_labels, top_values_np, color='skyblue')
    ax.set_xlabel('Top Probabilities')
    ax.set_ylabel('Classes')
    ax.set_title('Top 3 Maximum Possibilities and their Classes')

    return fig
            

