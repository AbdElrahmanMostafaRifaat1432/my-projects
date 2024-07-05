import streamlit as st
from PIL import Image
import io

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration



# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.set_page_config(page_title="Image Captioning", page_icon=":tada:", layout="wide")

# ---- IMAGE CAPTIONING SECTION ----
st.title("Image Captioning with BLIP Model")

# Load your image captioning model and other assets here
# This is just a placeholder for your model code
def dummy_image_captioning(image):

    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(pixel_values=inputs['pixel_values'] , max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]




    return generated_caption

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
left_column, right_column = st.columns((1, 1))
    

if uploaded_file is not None:

    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    with left_column:
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
    with right_column:
        
        st.write("Generating caption...")
        # Generate caption using your model
        caption = dummy_image_captioning(image)
        st.write(caption)
