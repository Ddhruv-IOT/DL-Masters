# import cv2

# def load_and_prep_image(frame, img_shape=48):
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#     gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = faceCascade.detectMultiScale(gray_img, 1.1, 4)

#     for x, y, w, h in faces:
#         roi_gray_img = gray_img[y:y + h, x:x + w]
#         roi_img = frame[y:y + h, x:x + w]

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         faces_roi = faceCascade.detectMultiScale(roi_img, 1.1, 4)

#         if len(faces_roi) == 0:
#             print("No Faces Detected")
#             # raise Exception("No Faces Detected")
#         else:
#             for (ex, ey, ew, eh) in faces_roi:
#                 img = roi_img[ey:ey + eh, ex:ex + ew]

#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             rgb_img = cv2.resize(rgb_img, (img_shape, img_shape))
#             rgb_img = rgb_img / 255.

#             return frame, rgb_img

# # Capture video from webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame, processed_img = load_and_prep_image(frame)

#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from mtcnn import MTCNN
import cv2

# Initialize MTCNN detector
detector = MTCNN()

def load_and_prep_image(frame, img_shape=48):
    # Detect faces using MTCNN
    result = detector.detect_faces(frame)

    for face_data in result:
        x, y, width, height = face_data['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Crop face region
        face_region = frame[y:y + height, x:x + width]

        # Resize and preprocess the face image
        face_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (img_shape, img_shape))
        face_image = face_image / 255.

        return frame, face_image

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, processed_img = load_and_prep_image(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
