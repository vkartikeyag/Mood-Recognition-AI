import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Mood Recognition AI", page_icon="😊")
st.title("🧠 Mood Recognition AI")
st.markdown("Take a photo and let AI detect your emotion!")

st.info("🔒 Privacy Notice: Photos are processed in real-time and never stored or saved anywhere. Your image is used solely for emotion detection and is discarded immediately after.")

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="trpakov/vit-face-expression")

img_file = st.camera_input("📷 Take a photo")

if img_file:
    with st.spinner("Analyzing your mood..."):
        classifier = load_model()
        img = Image.open(img_file)
        results = classifier(img)

    if results:
        top = results[0]
        dominant = top['label']
        confidence = top['score'] * 100
        emoji_map = {
            "happy": "😄", "sad": "😢", "angry": "😠",
            "fear": "😱", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
        }
        emoji = emoji_map.get(dominant.lower(), "🙂")
        st.success(f"{emoji} Mood: **{dominant.upper()}** ({confidence:.2f}%)")
        st.bar_chart({r['label']: r['score'] for r in results})
    else:
        st.warning("⚠️ No face detected. Try again!")
