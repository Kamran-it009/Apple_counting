import cv2
from PIL import Image
import streamlit as st
from Detector import YOLOV5_Detector
from tempfile import NamedTemporaryFile

detector = YOLOV5_Detector(weights='apples_detection.pt',
                           img_size=416,
                           confidence_thres=0.4,
                           iou_thresh=0.45,
                           agnostic_nms=True,
                           augment=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Counting Apples")

buffer = st.file_uploader("Upload Image File")
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    st.markdown('Upload complete!')
    st.markdown('''
            <style>
                .uploadedFile {display: none}
            <style>''',
                unsafe_allow_html=True)
    temp_file.write(buffer.getvalue())
    img = Image.open(temp_file.name)
    image = cv2.imread(temp_file.name)
    prediction, count = detector.Detect(image)
    rgb = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    st.image(rgb)
    st.write("Total Apples Count: " + str(count))
