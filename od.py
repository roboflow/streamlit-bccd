import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow


class ObjectDetectionApp:
    def __init__(self, workspace_id: str, model_id: str, version_number: str, private_api_key: str, 
                 confidence_threshold: str = '40', overlap_threshold: str = '30', include_bbox: str = 'Yes', 
                 include_class: str = 'Show Labels'):
        self.workspace_id = workspace_id
        self.model_id = model_id
        self.version_number = version_number
        self.private_api_key = private_api_key
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        self.include_bbox = include_bbox
        self.include_class = include_class

    def draw_boxes(self, model_object, img_object, uploaded_file: str, show_bbox: str, show_class_label: str,
                   font = cv2.FONT_HERSHEY_SIMPLEX):
        
        if isinstance(uploaded_file, str):
            img = cv2.imread(uploaded_file)
            predictions = model_object.predict(uploaded_file, confidence=int(self.confidence_threshold),
                                           overlap=self.overlap_threshold)
        else:
            predictions = model_object.predict(uploaded_file, confidence=int(self.confidence_threshold),
                                           overlap=self.overlap_threshold)
        predictions_json = predictions.json()
        for bounding_box in predictions:
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2
            class_name = bounding_box['class']
            confidence_score = bounding_box['confidence']
            box = (x0, x1, y0, y1)

            # position coordinates: start = (x0, y0), end = (x1, y1)
            # color = RGB-value for bounding box color, (0,0,0) is "black"
            # thickness = stroke width/thickness of bounding box
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    font,#font
                    0.4,#font scale
                    (255,255,255),#text color
                    thickness=1#thickness/"weight" of text
                    )
            color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
        
        return pil_image, predictions_json

    def run_inference(self):
        rf = Roboflow(api_key=self.private_api_key)
        project = rf.workspace(self.workspace_id).project(self.model_id)
        project_metadata = project.get_version_information()
        version = project.version(self.version_number)
        model = version.model

        for version_number in range(len(project_metadata)):
            try:
                if int(project_metadata[version_number]['model']['id'].split('/')[1]) == int(version.version):
                    project_endpoint = st.write(f"#### Inference Endpoint: {project_metadata[version_number]['model']['endpoint']}")
                    model_id = st.write(f"#### Model ID: {project_metadata[version_number]['model']['id']}")
                    version_name  = st.write(f"#### Version Name: {project_metadata[version_number]['name']}")
                    input_img_size = st.write(f"Input Image Size for Model Training (pixels, px):")
                    width_metric, height_metric = st.columns(2)
                    width_metric.metric(label='Pixel Width', value=project_metadata[version_number]['preprocessing']['resize']['width'])
                    height_metric.metric(label='Pixel Height', value=project_metadata[version_number]['preprocessing']['resize']['height'])

                    if project_metadata[version_number]['model']['fromScratch']:
                        train_checkpoint = 'Checkpoint'
                        st.write(f"#### Version trained from {train_checkpoint}")
                    elif project_metadata[version_number]['model']['fromScratch'] is False:
                        train_checkpoint = 'Scratch'
                        train_checkpoint = st.write(f"#### Version trained from {train_checkpoint}")
                    else:
                        train_checkpoint = 'Not Yet Trained'
                        train_checkpoint = st.write(f"#### Version is {train_checkpoint}")
            except KeyError:
                continue

        ## Subtitle.
        st.write('### Inferenced/Prediction Image')
        
        ## Pull in default image or user-selected image.
        if uploaded_file is None:
            # Default image.
            default_img_path = "images/fast-break5.jpg"
            image = Image.open(default_img_path)
            original_image = image
            open_cv_image = cv2.imread(default_img_path)
            original_opencv_image = open_cv_image
            # Display response image.
            pil_image_drawBoxes, json_values = self.draw_boxes(model, default_img_path, default_img_path,
                                                                    st.session_state['include_bbox'],
                                                                    st.session_state['include_class'])
            
        else:
            # User-selected image.
            image = Image.open(uploaded_file)
            original_image = image
            opencv_convert = image.convert('RGB')
            open_cv_image = np.array(opencv_convert)
            # Convert RGB to BGR: OpenCV deals with BGR images rather than RGB
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            # Convert PIL image to byte-string so it can be sent for prediction to the Roboflow Python Package
            b = io.BytesIO()
            image.save(b, format='JPEG')
            im_bytes = b.getvalue() 
            # Display response image.
            pil_image_drawBoxes, json_values = self.draw_boxes(model, open_cv_image, im_bytes,
                                                                    st.session_state['include_bbox'],
                                                                    st.session_state['include_class'])
        
        st.image(pil_image_drawBoxes,
                use_column_width=True)
        # Display original image.
        st.write("#### Original Image")
        st.image(original_image,
                use_column_width=True)

    # more class methods go here

##########
##### Set up sidebar.
##########
image = Image.open("./images/roboflow_logo.png")
st.sidebar.image(image,
                use_column_width=True)

# Add in location to select image.
st.sidebar.write("#### Select an image to upload.")
uploaded_file = st.sidebar.file_uploader("",
                                        type=['png', 'jpg', 'jpeg'],
                                        accept_multiple_files=False)

st.sidebar.write("[Find additional images on Roboflow Universe.](https://universe.roboflow.com/)")
st.sidebar.write("[Improving Your Model with Active Learning](https://help.roboflow.com/implementing-active-learning)")

## Add in sliders.
show_bbox = st.sidebar.radio("Show Bounding Boxes:",
                            options=['Yes', 'No'],
                            key='include_bbox')

show_class_label = st.sidebar.radio("Show Class Labels:",
                                    options=['Show Labels', 'Hide Labels'],
                                    key='include_class')

confidence_threshold = st.sidebar.slider("Confidence threshold (%): What is the minimum acceptable confidence level for displaying a bounding box?", 0, 100, 40, 1)
overlap_threshold = st.sidebar.slider("Overlap threshold (%): What is the maximum amount of overlap permitted between visible bounding boxes?", 0, 100, 30, 1)

##########
##### Set up project access.
##########

## Title.
st.write("# Roboflow Object Detection Tests")

with st.form("project_access"):
    workspace_id = st.text_input('Workspace ID', key='workspace_id',
                                help='Finding Your Project Information: https://docs.roboflow.com/python#finding-your-project-information-manually',
                                placeholder='Input Workspace ID')
    model_id = st.text_input('Model ID', key='model_id', placeholder='Input Model ID')
    version_number = st.text_input('Trained Model Version Number', key='version_number', placeholder='Input Trained Model Version Number')
    private_api_key = st.text_input('Private API Key', key='private_api_key', type='password',placeholder='Input Private API Key')
    submitted = st.form_submit_button("Verify and Load Model")

if submitted:
    st.write("Loading model...")
    app = ObjectDetectionApp(workspace_id, model_id, version_number, private_api_key)
    app.run_inference()
