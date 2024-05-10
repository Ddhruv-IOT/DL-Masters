import time
import numpy as np
import cv2

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

def load_and_prep_image(filename, img_shape = 48):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  

    img = filename

    GrayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)
    
    for x,y,w,h in faces:
        
        roi_GrayImg = GrayImg[ y: y + h , x: x + w ]
        roi_Img = img[ y: y + h , x: x + w ]
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        faces = faceCascade.detectMultiScale(roi_Img, 1.1, 4)
       
        if len(faces) == 0:
            print("No Faces Detected")
            raise Exception("No Faces Detected")
        
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ ey: ey+eh , ex: ex+ew ]
    
            RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            RGBImg = cv2.resize(RGBImg,(img_shape,img_shape))

            RGBImg = RGBImg/255.

            return RGBImg

def pred_and_plot(filename, class_names, size, cnn_loaded_model, tf):

    try: 
        img = load_and_prep_image(filename, size)
        
        # Make a prediction
        # Model_Prediction = np.argmax(cnn_loaded_model.predict(tf.expand_dims(img, axis=0), verbose=0))
        # pred_class = class_names[Model_Prediction]
        # print("Sentiment Identified as: ", pred_class)
        # return img, pred_class
        predictions = cnn_loaded_model.predict(tf.expand_dims(img, axis=0), verbose=0)
        Model_Prediction = np.argmax(predictions)
        confidence = predictions[0, Model_Prediction]  # Confidence value for the predicted class
        pred_class = class_names[Model_Prediction]
        print("Sentiment Identified as: ", pred_class)
        print("Confidence:", confidence)
        return img, pred_class, confidence
    
    except Exception as e:
        raise Exception(e)


def analysize(img_buffer, model, st, tf):
    bytes_data = img_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    try: 
        img, em, cf  = pred_and_plot(cv2_img, ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], 48, model, tf)
        return img, em, cf
    except Exception as e:
        st.error("No Faces Detected. Please try again with a clear selfie.")