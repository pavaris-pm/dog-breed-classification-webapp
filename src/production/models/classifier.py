from transformers import AutoFeatureExtractor, AutoModelForImageClassification  
from typing import List, Tuple, Dict
from .convnext.core import ConvNextTransfer
from .convnext.utils import get_total_class
from torch import nn
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import torch



def load_classifier_model()->Tuple[AutoFeatureExtractor, AutoModelForImageClassification]:
    # load convnext from huggingface hub
    extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224-22k")
    convnext = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224-22k")
    return (extractor, convnext)

def init_model(weight_path: str)->nn.Module:
    # initilialize a model
    extractor , convnext = load_classifier_model()

    conv_block = convnext.convnext
    predictor = ConvNextTransfer(conv_block)

    if weight_path:
        # Load the model's state dictionary
        state_dict = torch.load(weight_path)

        # Load the state dictionary to the model
        predictor.load_state_dict(state_dict)
        return predictor, extractor

    else:
        # in case that there has no weight to be loaded
        return predictor, extractor
    
def classify(image: Image.Image)->Tuple[torch.Tensor, 
                                        torch.Tensor,
                                        str
                                        ]:
    # init this by ourselves
    weight_file = None# '/workspaces/dog-breed-classification-webapp/src/production/models/model_weights/best_model_convnext.pth'
    _ , idx2label = get_total_class()

    # init the model
    classifier, extractor = init_model(weight_file)

    # feature extraction
    inputs = extractor(image, return_tensors="pt")
    raw_logits = classifier(inputs.pixel_values)
    probs = torch.softmax(raw_logits, dim=1)
    # this will predict as a idx
    prediction = torch.softmax(raw_logits, dim=1).argmax(dim=1)

    cls_prediction = idx2label[prediction.item()]

    return (probs, prediction, cls_prediction)


def plot_probs_distribution(probs: torch.Tensor)->None:

    # Get the top 5 maximum values and indices along dimension 1
    top_values, top_indices = torch.topk(probs, k=3, dim=1)

    # Convert tensors to numpy arrays for plotting
    top_values_np = top_values.squeeze().detach().numpy()
    top_indices_np = top_indices.squeeze().detach().numpy()

    labels = ['a', 'b', 'c']

    # Create the plot using Matplotlib
    fig, ax = plt.subplots()
    ax.barh(labels, top_values_np, color='skyblue')
    ax.set_xlabel('Top Values')
    ax.set_ylabel('Indices')
    ax.set_title('Top 5 Maximum Values and their Indices')

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)

    return None
            

