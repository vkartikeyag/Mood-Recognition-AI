import streamlit as st
from fer import FER
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mood Recognition AI", page_icon="😊")

st.title("🧠 Mood Recognition AI")
st.markdown("Take a photo and let AI detect your emotion in real-time!")

img_file = st.camera_input("📷 Take a photo for mood detection")

if img_file:
    with st.spinner("Analyzing your mood..."):
        img = Image.open(img_file)
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detector = FER(mtcnn=False)
        result = detector.detect_emotions(frame_bgr)

    if result:
        emotions = result[0]['emotions']
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant] * 100

        emoji_map = {
            "happy": "😄", "sad": "😢", "angry": "😠",
            "fear": "😱", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
        }
        emoji = emoji_map.get(dominant, "🙂")

        st.success(f"{emoji} Detected Mood: **{dominant.upper()}** ({confidence:.2f}%)")
        st.markdown("### All Emotion Scores")
        st.bar_chart(emotions)
    else:
        st.warning("⚠️ No face detected. Please try again with better lighting!")
```

---

**`requirements.txt`**
```
streamlit
fer
opencv-python-headless
pillow
numpy
tensorflow-cpu
```

---

**`packages.txt`**
```
libgl1
