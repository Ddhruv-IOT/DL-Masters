import numpy as np
import cv2
from mtcnn import MTCNN
import time
import pandas as pd
import keyring


detector = MTCNN()

def typewriter(text: str, speed: int, st: object, init_state: bool = False) -> int:
    container = st.empty()

    if init_state:
        for index in range(len(text) + 1):
            curr_partial_text = text[:index]
            container.markdown(curr_partial_text)
            time.sleep(0.1 / speed)
    container.markdown(text)
    return 0

def take_selfie(camera_container, update_state_cb, key):
     img_file_buffer = camera_container.camera_input(":green[Analyse your mood with a selfie!]", disabled=False, on_change=update_state_cb,key=key)
     return img_file_buffer

def display_pic(camera_container, key, st, update_state_cb):
    
    img_file_buffer = take_selfie(camera_container, update_state_cb, key)
    
    if st.session_state.img_clicked and img_file_buffer is not None:
        camera_container.empty()
        with camera_container.expander("Your Selfie"):
            st.image(img_file_buffer, use_column_width=True)
        bytes_data = img_file_buffer
        return bytes_data
    return None

def img_buffer_to_cv2_img(img_buffer):
    bytes_data = img_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return cv2_img

def find_face(frame):
    # Detect faces using MTCNN
    result = detector.detect_faces(frame)

    for face_data in result:
        x, y, width, height = face_data['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Crop face region
        face_region = frame[y:y + height, x:x + width]

        # Resize and preprocess the face image
        face_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        return frame, face_image

def resize_image(face_image, img_shape=48):
    face_image = cv2.resize(face_image, (img_shape, img_shape))
    face_image = face_image / 255.
    return face_image

def analyze_emotions(img, cnn_loaded_model, tf):
    class_names =  ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    try: 
        
        # Make a prediction
        Model_Prediction = np.argmax(cnn_loaded_model.predict(tf.expand_dims(img, axis=0), verbose=0))
        pred_class = class_names[Model_Prediction]
        
        print("Sentiment Identified as: ", pred_class)
        
        predictions = cnn_loaded_model.predict(tf.expand_dims(img, axis=0), verbose=0)
        Model_Prediction = np.argmax(predictions)
        confidence = predictions[0, Model_Prediction]  # Confidence value for the predicted class
        pred_class = class_names[Model_Prediction]
        
        print("Sentiment Identified as: ", pred_class)
        print("Confidence:", confidence)
        
        return img, pred_class, confidence

    except Exception as e:
        raise Exception(e)

def init_music_player():
    Music_Player = pd.read_csv("../big_data/spotify/data_moods.csv")
    Music_Player = Music_Player[['name','artist','mood','popularity']]
    return Music_Player

# Making Songs Recommendations Based on Predicted Class
def recommend_songs(pred_class, Music_Player):
    
    if( pred_class=='Sad' ):

        Play = Music_Player[Music_Player['mood'] =='Sad' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)

    if( pred_class=='Happy' or pred_class=='Neutral' ):

        Play = Music_Player[Music_Player['mood'] =='Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)

    if( pred_class=='Fear' or pred_class=='Angry' ):

        Play = Music_Player[Music_Player['mood'] =='Calm' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)

    if( pred_class=='Surprise' or pred_class=='Disgust' ):

        Play = Music_Player[Music_Player['mood'] =='Energetic' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
    
    return Play

def select_true_emotion(df):
    # Find the mode (most frequent emotion)
    mode_emotion = df['Predicted emotion'].mode()
    
    # If there's only one mode or no mode (all different emotions), return the emotion with the highest confidence
    if len(mode_emotion) == 1:
        mode_confidence = df[df['Predicted emotion'] == mode_emotion[0]]['Confidence'].max()
        return mode_emotion[0], mode_confidence
        # return mode_emotion[0]
    
    # Group by predicted emotion and find the row with the highest confidence for each emotion
    max_confidence_df = df.groupby('Predicted emotion')['Confidence'].max().reset_index()
    
    # Find the row with the highest confidence
    max_confidence_row = max_confidence_df.loc[max_confidence_df['Confidence'].idxmax()]

    unique_emotions = df['Predicted emotion'].unique()
    # print(unique_emotions)
    unique_confidences = df['Confidence'].unique()
    # print(unique_confidences)
    if len(unique_emotions) == 4 and  len(unique_confidences) == 1:
        return "Tie", 100
        
    return max_confidence_row['Predicted emotion'], max_confidence_row['Confidence']


def read_credential(service_name, username):
    return keyring.get_password(service_name, username)
