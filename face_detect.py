import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('C:\\Users\\harsh\\PycharmProjects\\haarcascade_frontalface_default.xml')

#img = cv2.imread('C:\\Users\\harsh\\Desktop\\rdj.png') 
webcam = cv2.VideoCapture(0)

while True:
     
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow('Harsh face detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
# After the loop release the cap object
webcam.release()
# Destroy all the windows
cv2.destroyAllWindows()



