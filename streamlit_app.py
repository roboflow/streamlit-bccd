import streamlit as st
import requests
import base64
import io
from statistics import mean
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roboflow import Roboflow

##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write("#### Select an image to upload.")
uploaded_file = st.sidebar.file_uploader("",
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

st.sidebar.write("[Find additional images on Roboflow Universe.](https://universe.roboflow.com/)")

## Add in sliders.
confidence_threshold = st.sidebar.slider("Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?", 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider("Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?", 0.0, 1.0, 0.5, 0.01)


image = Image.open("./images/roboflow_logo.png")
st.sidebar.image(image,
                 use_column_width=True)

image = Image.open("./images/streamlit_logo.png")
st.sidebar.image(image,
                 use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write("# Roboflow Object Detection Tests")

rf = Roboflow(api_key=f"{st.secrets['api_key']}")
project = rf.workspace("mohamed-traore-2ekkp").project("boxes-on-a-conveyer-belt")
project_metadata = project.get_version_information()
# dataset = project.version(5).download("yolov5")
version = project.version(5)
model = version.model

project_type = st.write(f"#### Project Type: {project.type}")
for version_number in range(len(project_metadata)):
    if int(project_metadata[version_number]['model']['id'].split('/')[1]) = int(version.version):
      project_endpoint = st.write(f"#### Inference Endpoint: {project_metadata[version_number]['model']['endpoint']}")

model_id = st.write(f"#### Model ID: {project_metadata['model']['id']}")
version_name  = st.write(f"#### Version Name: {project_metadata['name']}")
input_img_size = st.write(f"Input Image Size (pixels, px):")
width_metric, height_metric = st.column(2)
width_metric.metric(label='Pixel Width', value=project_metadata['preprocessing']['resize']['width'])
height_metric.metric(label='Pixel Height', value=project_metadata['preprocessing']['resize']['height'])

if project_metadata[version_number]['model']['fromScratch']:
  train_checkpoint = 'Checkpoint'
  st.write(f"#### Version trained from {train_checkpoint}")
elif project_metadata[version_number]['model']['fromScratch'] is False:
  train_checkpoint = 'Scratch'
  train_checkpoint = st.write(f"#### Version trained from {train_checkpoint}")
else:
  train_checkpoint = 'Not Yet Trained'
  train_checkpoint = st.write(f"#### Version is {train_checkpoint}")

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    default_img_path = "images/test_box.jpg"
    image = Image.open(default_img_path)

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

## Construct the URL to retrieve image.
upload_url = ''.join([
    'https://detect.roboflow.com/boxes-on-a-conveyer-belt/3',
    f"?api_key={st.secrets['api_key']}",
    '&format=image',
    f'&overlap={overlap_threshold * 100}',
    f'&confidence={confidence_threshold * 100}',
    '&stroke=4',
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

# Display response image.
st.image(image,
         use_column_width=True)
# Display original image.
st.write("#### Original Image")
st.image(original_image,
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
## Generate list of classes.
class_list = [box['class'] for box in output_dict['predictions']]

json_tab, statistics_tab, project_tab = st.tabs(["JSON Output", "Prediction Statistics", "Project Info"])

with json_tab:
  ## Display the JSON in main app.
  st.write('### JSON Output')
  st.write(r.json())

with statistics_tab:
  ## Summary statistics section in main app.
  st.write('### Summary Statistics')
  with st.column():
    st.metric(label=f'Number of Bounding Boxes (ignoring overlap thresholds)', value='{len(confidences)}')
    st.metric(label=f'Average Confidence Level of Bounding Boxes:', value='{(np.round(np.mean(confidences),4))}')

  ## Histogram in main app.
  st.write('### Histogram of Confidence Levels')
  fig, ax = plt.subplots()
  ax.hist(confidences, bins=10, range=(0.0,1.0))
  st.pyplot(fig)

  ## Dataframe in main app with confidence level by class
  predictions_df = pd.DataFrame(list(zip(class_list, confidences)), columns = ['Class', 'Confidence'])
  st.dataframe(predictions_df)

with project_tab:
  col1, col2, col3 = st.columns(3)
  for version_number in range(len(project_metadata)):
    if int(project_metadata[version_number]['model']['id'].split('/')[1]) = int(version.version):
      col1.write(f'Total images in the version: {version.images}')
      col1.metric(label='Augmented Train Set Image Count', value=version.splits['train'])
      col2.metric(label='mean Average Precision (mAP)', value=f"{project_metadata[version_number]['model']['map']:2%}")
      col2.metric(label='Precision', value=f"{project_metadata[version_number]['model']['precision']:2%}")
      col2.metric(label='Recall', value=f"{project_metadata[version_number]['model']['recall']:2%}")
      col3.metric(label='Train Set Image Count', value=project.splits['train'])
      col3.metric(label='Valid Set Image Count', value=project.splits['valid'])
      col3.metric(label='Test Set Image Count', value=project.splits['test'])

  col4, col5, col6 = st.columns(3)
  col4.write('Preprocessing steps applied:')
  col4.json(version.preprocessing)
  col5.write('Augmentation steps applied:')
  col5.json(version.augmentation)
  col6.metric(label='Train Set', value=version.splits['train'], delta=f"{(version.splits['train'] / project.splits['train']):.2%)}")
  col6.metric(label='Valid Set', value=version.splits['valid'], delta=f"{(version.splits['valid'] / project.splits['valid']):.2%)}")
  col6.metric(label='Test Set', value=version.splits['test'], delta=f"{(version.splits['test'] / project.splits['test']):.2%}")
  
  col7, col8, col9 = st.columns(3)
  col7.write(f"Annotation Group Name: {project.annotation}")
