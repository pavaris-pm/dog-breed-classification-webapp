from torch import nn
import torch
from PIL import Image
import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection

@st.cache_resource
# to load a DETR as a detector model
def load_detector_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return processor, detector

def detect(image: Image):
    processor, detector = load_detector_model()
    inputs = processor(images=image, return_tensors="pt")
    outputs = detector(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        prediction = detector.config.id2label[label.item()]
        return (
                f"Detected {prediction} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        ), prediction, box
