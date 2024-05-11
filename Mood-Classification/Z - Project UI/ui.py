from utils import typewriter, take_selfie, display_pic, analyze_emotions, find_face, resize_image, img_buffer_to_cv2_img 
from utils import init_music_player, recommend_songs
from utils import select_true_emotion
import streamlit as st
import pandas as pd
import tensorflow as tf
import time 

print = st.write

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
                frame, face = find_face(img_bytes)
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
        
        with st.spinner("Analysing Emotion..."):
            for i, model in enumerate(models):
                if i == 0:
                    img = resize_image(face, 48)
                else:
                    img = resize_image(face, 224)
                    
                img, emotion, confidence = analyze_emotions(img, model, tf)
                confidence = round(confidence * 100, 2)
                predicted_emotion.append(emotion)
                predicted_confidence.append(confidence)
                
            data = {
                'Model name': model_names,
                'Predicted emotion': predicted_emotion,
                'Confidence': predicted_confidence
                }

            df = pd.DataFrame(data)
            st.header("Emotion Analysis Summary")
            st.table(df)
            
            emotion, confidence = select_true_emotion(df)

            if emotion not in ["Angry", "Disgust", "Fear", "Sad"]:
                st.write(f"#### The detected Emotion is :green[{emotion}], {confidence}% confident")
            else:
                st.write(f"#### The detected Emotion is :red[{emotion}], {confidence}% confident")
        
        with st.spinner("Recommending Songs..."):                
            st.header("Recommended Songs")
            music_palyer = init_music_player()
            music_list = recommend_songs(emotion, music_palyer)
            # confidence to be fixed
            st.table(music_list)
        

with st.sidebar:
    st.title("FeelSonic")
    st.write("Your Emotion-Driven Music Companion")
    if st.session_state.img_clicked != False:
        if st.button("Retake Selfie"):
            camera_container.empty()
            st.session_state.img_clicked = False
            st.rerun()




