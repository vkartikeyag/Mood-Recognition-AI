import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

st.title("AI Mood Detection System")

img_file = st.camera_input("Take a photo for mood detection")

if img_file:
    img = Image.open(img_file)
    frame = np.array(img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        result = DeepFace.analyze(frame_bgr, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion]

        st.success(f"Mood: **{emotion}** ({confidence:.2f}%)")

    except:
        st.error("No face detected. Try again!")