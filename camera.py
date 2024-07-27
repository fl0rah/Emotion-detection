import cv2
import numpy as np
import tensorflow as tf



emotion_nam = {0: 'Angry', 1: 'Disgust', 2: 'Fear',3: 'Happy', 4:'Neutral' , 5:'Sad', 6:'Surprise'}
model_path = 'C:/Users/Desktop/cam/photo_archive/Emotion_model.h5'
model = tf.keras.models.load_model(model_path)

vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        gray_face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(gray_face, (48, 48))
        resized_face = np.expand_dims(np.expand_dims(resized_face, -1), 0)
        emotion_pred = model.predict(resized_face)
        maxindex = int(np.argmax(emotion_pred))
        
        emotion_name = emotion_nam[maxindex]
        cv2.putText(frame, emotion_name, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Camera_of_Emotions', frame)

    if cv2.waitKey(1) & 0xFF == ord('o'):
        cv2.imwrite('emotion_cam.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
vid.release()
cv2.destroyAllWindows()