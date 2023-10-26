from typing import List, Tuple, Dict
import streamlit as st
from PIL import Image, ImageDraw


# to read an image for overlaying
def read_image(image):
    img = Image.open(image)
    return img

# to overlay a bounding box to an image
def bbox_overlay(img: Image, prediction: str, bbox: List[float])->None:

    # init pillow drawing object 
    draw = ImageDraw.Draw(img)

    # get a position that object were detected
    x, y, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    draw.rectangle([x, y, x + width, y + height], outline=(255, 0, 0), width=5)
    
    st.title("Image with Bounding Boxes")

    # Add a smaller text as a description within the title using markdown
    st.markdown(f"### Predicted Class: {prediction}")

    # Change the font size using custom CSS
    st.markdown(
        """
        <style>
        h3 {
            font-size: 20px;
            color: #FF5733; /* Change the color if needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the image with bounding boxes in Streamlit
    st.image(img, channels="BGR", caption="Image with Bounding Boxes", use_column_width=True)

    return None
