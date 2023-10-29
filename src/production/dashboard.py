import streamlit as st
from models.detector import detect
from models.classifier import classify, plot_probs_distribution
from utils import read_image, bbox_overlay


# construct a dashboard gui
st.title("Dog Breed Classification 🐶 🎖️")
# Add a smaller text as a description within the title using markdown
st.markdown(f"### Developed by Mr. Pavaris Ruangchutiphophan")

# Change the font size using custom CSS
st.markdown(
    """
    <style>
    h3 {
        font-size: 20px;
        color: #333; /* Change the color if needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
upload_img = st.file_uploader(label='Upload Image:', type=["png", "jpg", "jpeg"])

if upload_img:
    img = read_image(upload_img)
    txt_result, prediction, bbox = detect(img)
    probs, _ , cls_prediction = classify(img)
    # to display text of prediction
    # st.text(txt_result) # to display confidence score and bbox
    # st.text(f'ConvNext Prediction: {cls_prediction}')
    # to overlay bounding box onto an image
    # then pass the convnext prediction
    bbox_overlay(img, cls_prediction, bbox)
    fig = plot_probs_distribution(probs)
    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)


    


