import streamlit as st
from PIL import Image, ImageOps
from models.detector import detect

# construct a dashboard gui
st.title("Dog Breed Classification")
upload_img = st.file_uploader(label='Upload Image:', type=["png", "jpg", "jpeg"])

if upload_img:
    img = Image.open(upload_img)
    result = detect(img)
    st.text(result)


