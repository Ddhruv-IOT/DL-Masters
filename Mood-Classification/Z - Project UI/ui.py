from utils import typewriter, take_selfie, display_pic, pred_and_plot, analysize

import streamlit as st
import pandas as pd
import tensorflow as tf


cnn_loaded_model = tf.keras.models.load_model("./model_nn.h5")

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
        with st.spinner():
            # TODO: Add Tru Catch
            img, em, cf =  analysize(img_buffer, cnn_loaded_model, st, tf)
            # TODO: ADD Colors here
            st.write(f"The detected Emotion is :green[{em}], {cf}")

with st.sidebar:
    st.title("FeelSonic")
    st.write("Your Emotion-Driven Music Companion")
    if st.session_state.img_clicked != False:
        if st.button("Retake Selfie"):
            camera_container.empty()
            st.session_state.img_clicked = False
            st.rerun()