from typing import List, Tuple, Dict
import streamlit as st
from PIL import Image, ImageDraw
import glob


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

# path checking in file directory
# count data in each fold in each categories
def get_total_class()->Tuple[Dict, Dict]:
  # to return label2idx and idx2label in order to match simplt with number for preds
  c = 0
  tot_classes = {}
  for img_path in glob.glob('/workspaces/dog-breed-classification-webapp/dog-breeds-classification/dogImages/train/*/*'):
    # to get breed type and remove number out from breed name
    breed_type = img_path.split('/')[-2][4:]
    if breed_type not in tot_classes.keys():
      tot_classes[breed_type] = c
      c += 1

  # convert the label back
  reverse_tot_classes = {v:k for k, v in tot_classes.items()}
  return (tot_classes, reverse_tot_classes)



def visualize_breed_fold(fold: str)->Dict:
  count_stats = {}
  for img_path in glob.glob(f'/workspaces/dog-breed-classification-webapp/dog-breeds-classification/dogImages/{fold}/*/*'):
    # to get breed type and remove number out from breed name
    breed_type = img_path.split('/')[-2][4:]
    if breed_type not in count_stats.keys():
      count_stats[breed_type] = 1
    else:
      count_stats[breed_type] += 1

  return count_stats
