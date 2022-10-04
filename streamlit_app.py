import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow

##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

st.sidebar.write('[Find additional images on Roboflow.](https://universe.roboflow.com/)')

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)


image = Image.open('./images/roboflow_logo.png')
st.sidebar.image(image,
                 use_column_width=True)

image = Image.open('./images/streamlit_logo.png')
st.sidebar.image(image,
                 use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write('# Roboflow Object Detection Tests')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)

## Subtitle.
st.write('### Inferenced Image')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

rf = Roboflow(api_key=f"{st.secrets['api_key']}")
project = rf.workspace("mohamed-traore-2ekkp").project("boxes-on-a-conveyer-belt")
# dataset = project.version(5).download("yolov5")
version = project.version(5)
model = version.model

## Construct the URL to retrieve image.
upload_url = ''.join([
    'https://detect.roboflow.com/boxes-on-a-conveyer-belt/3',
    f"?api_key={st.secrets['api_key']}",
    '&format=image',
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&stroke=2',
    '&labels=True'
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

image = Image.open(BytesIO(r.content))

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Display image.
st.image(image,
         use_column_width=True)

## Construct the URL to retrieve JSON.
upload_url = ''.join([
    'https://detect.roboflow.com/boxes-on-a-conveyer-belt/3',
    f"?api_key={st.secrets['api_key']}"
])

## POST to the API.
r = requests.post(upload_url,
                  data=img_str,
                  headers={
    'Content-Type': 'application/x-www-form-urlencoded'
})

## Save the JSON.
output_dict = r.json()

## Generate list of confidences.
confidences = [box['confidence'] for box in output_dict['predictions']]

## Summary statistics section in main app.
st.write('### Summary Statistics')
st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

## Histogram in main app.
st.write('### Histogram of Confidence Levels')
fig, ax = plt.subplots()
ax.hist(confidences, bins=10, range=(0.0,1.0))
st.pyplot(fig)

## Display the JSON in main app.
st.write('### JSON Output')
st.write(r.json())

col1, col2, col3 = st.columns(3)
col1.metric(label='Project Type', value=project.type)
col2.metric(label='mean Average Precision (mAP)', value='- %')
col2.metric(label='Precision', value='- %')
col2.metric(label='Recall', value='- %')
col3.metric(label='Train Set', value=project.splits['train'])
col3.metric(label='Valid Set', value=project.splits['valid'])
col3.metric(label='Test Set', value=project.splits['test'])

col4, col5, col6 = st.columns(3)
col4.write(f'Total images in the version: {version.images}')
col5.write('Preprocessing steps applied:')
col5.json(version.preprocessing)
col5.write('Augmentation steps applied:')
col5.json(version.augmentation)
col5.metric(label='Augmented Train Set', value=version.splits['train'])
col6.metric(label='Train Set', value=version.splits['train'], delta=f"{((version.splits['train'] / project.splits['train'])*100)}%")
col6.metric(label='Valid Set', value=version.splits['valid'], delta=f"{((version.splits['valid'] / project.splits['valid'])*100)}%")
col6.metric(label='Test Set', value=version.splits['test'], delta=f"{((version.splits['test'] / project.splits['test'])*100)}%")
