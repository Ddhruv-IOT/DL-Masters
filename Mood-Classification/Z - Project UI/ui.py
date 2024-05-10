from utils import typewriter, take_selfie, display_pic, analyze_emotions, find_face_and_resize, img_buffer_to_cv2_img 
import streamlit as st
import pandas as pd
import tensorflow as tf
import time 


# cnn_loaded_model = tf.keras.models.load_model("./big_data/Saved_Models/sentiment_CNN_model.h5")
# mbn_loaded_model = tf.keras.models.load_model("./big_data/Saved_Models/sentiment_MobileNet_model.h5")
# rsn_loaded_model = tf.keras.models.load_model("./big_data/Saved_Models/sentiment_ResNet_model.h5")
# vgg_loaded_model = tf.keras.models.load_model("./big_data/Saved_Models/sentiment_VGG19_model.h5")

# models = [cnn_loaded_model, mbn_loaded_model, rsn_loaded_model, vgg_loaded_model]
# model_names = ["CNN", "MobileNet", "ResNet", "VGG19"]

# Load models
model_paths = [
    "../big_data/Saved_Models/sentiment_CNN_model.h5",
    "../big_data/Saved_Models/sentiment_MobileNet_model.h5",
    "../big_data/Saved_Models/sentiment_ResNet_model.h5",
    "../big_data/Saved_Models/sentiment_VGG19_model.h5"
]

model_names = ["CNN", "MobileNet", "ResNet", "VGG19"]
models = [tf.keras.models.load_model(path) for path in model_paths]

predicted_emotion = []
predicted_confidence = []

if 'welcome_init_state' not in st.session_state:
	st.session_state.welcome_init_state = True

if 'img_clicked' not in st.session_state:
	st.session_state.img_clicked = False
      
def update_state_cb():
	st.session_state.img_clicked = True

is_completed = typewriter(text="# Hi User!", speed=5, st=st, init_state=st.session_state.welcome_init_state)


if (is_completed == 0):
    st.session_state.welcome_init_state = False
    
    st.subheader("What's on your mood today?")
    
    camera_container = st.empty()    
    img_buffer = display_pic(camera_container, 42, st, update_state_cb)

    if img_buffer:
        with st.spinner("Looking for Human Face..."):
            img_bytes = img_buffer_to_cv2_img(img_buffer)
            try: 
                frame, face = find_face_and_resize(img_bytes, 48)
            except TypeError:
                st.session_state.img_clicked = False
                camera_container.empty()
                st.session_state.img_clicked = False
                st.error("No face detected, please try again")
                st.header("Please try again, Restarting...")
                timer = st.empty()
                for i in range(3, 0, -1):
                    timer.text(f"Restarting in {i} seconds...")
                    camera_container.empty()
                    time.sleep(1)
                st.rerun()
                    
            with st.expander("Detected Face"):
                st.image(frame, use_column_width=True)
        
        # with st.spinner("Analysing Emotion..."):
        #     for model in models:
        #         img, emotion, confidence = analyze_emotions(face, model, tf)
        #         predicted_emotion.append(emotion)
        #         predicted_confidence.append(confidence)
            # TODO: ADD Colors here
            # confidence = round(confidence * 100, 2)
            # st.write(f"The detected Emotion is :green[{emotion}], {confidence}% confident")
        

with st.sidebar:
    st.title("FeelSonic")
    st.write("Your Emotion-Driven Music Companion")
    if st.session_state.img_clicked != False:
        if st.button("Retake Selfie"):
            camera_container.empty()
            st.session_state.img_clicked = False
            st.rerun()