import streamlit as st
import helper
import config
import numpy as np
from jina import Document


def show_pet_and_breed(tags, image):
    
    breed = tags['label']  #
    pet_category = 'cat' if breed[0].isupper() else 'dog'
    breed = breed.lower()
    breed = ' '.join(breed.split('_'))
    article = 'an' if breed[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
    st.image(image, caption="I am {} percent sure this is {} {} {}".format(
        round(tags['prob']*100), article, breed, pet_category))


# UI layout
st.set_page_config(page_title="Jina Pet Breed Classification")
st.markdown(
    body=helper.UI.css,
    unsafe_allow_html=True,
)
# Sidebar
st.sidebar.markdown(helper.UI.about_block, unsafe_allow_html=True)

# Title
st.header("Jina Pet Breed Classification")

# File uploader
upload_cell, preview_cell = st.columns([12, 1])
query = upload_cell.file_uploader("")

# If file is uploaded
if query:
    # if clicked on 
    if st.button(label="Classify"):
        # get tags (predicted breed and probability) for the given pet image
        tags, image = helper.get_breed(
            query, host=config.HOST, protocol=config.PROTOCOL, port=config.PORT)
        show_pet_and_breed(tags, image)
