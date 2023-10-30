import streamlit as st
from models.detector import load_detector_model, detect
from models.classifier import load_classifier_model, classify, plot_probs_distribution
from utils import read_image, bbox_overlay


# construct a dashboard gui
st.title("Dog Breed Classification 🐶 🎖️")
# Add a smaller text as a description within the title using markdown
st.markdown(f"### Developed by Mr. Pavaris Ruangchutiphophan")
st.text("Upload your image, it will return location of dog in image input and their breed")


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

# everything will be downloaded before making an inference
processor, detector = load_detector_model()
convnext, extractor = load_classifier_model('convnext_v2')

# to upload input image file
upload_img = st.file_uploader(label='Upload Image:', type=["png", "jpg", "jpeg"])



if upload_img:
    img = read_image(upload_img)
    txt_result, prediction, bbox = detect(img,
                                          processor,
                                          detector,
                                          )
    
    probs, _ , cls_prediction = classify(img,
                                         convnext,
                                         extractor,
                                         )
    # to display text of prediction
    # st.text(txt_result) # to display confidence score and bbox
    # st.text(f'ConvNext Prediction: {cls_prediction}')
    # to overlay bounding box onto an image
    # then pass the convnext prediction
    bbox_overlay(img, cls_prediction, bbox)
    fig = plot_probs_distribution(convnext, probs)
    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)


    


