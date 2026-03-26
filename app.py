import streamlit as st
from fer import FER
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mood Recognition AI", page_icon="😊")

st.title("😊 Mood Recognition AI")
st.write("Take a photo and the AI will detect your emotion!")

img_file = st.camera_input("📸 Click to take a photo")

if img_file:
    img = Image.open(img_file)
    frame = np.array(img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing your mood..."):
        detector = FER(mtcnn=False)
        result = detector.detect_emotions(frame_bgr)

    if result:
        emotions = result[0]['emotions']
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant] * 100

        st.success(f"**Detected Mood: {dominant.upper()}** ({confidence:.2f}%)")
        st.subheader("All Emotion Scores:")
        st.bar_chart(emotions)
    else:
        st.warning("😕 No face detected. Please try again with better lighting!")
```

---

**2. `requirements.txt`**
```
streamlit
fer
opencv-python-headless
pillow
numpy
tensorflow-cpu
```

---

**3. `packages.txt`**
```
libgl1
