import cv2
import os

import face_recognition

imagesPath = "C:/Users/Lyria/Documents/raquel_ia/faces"
facesEncoding = []
facesNames = []
for file_name in os.listdir(imagesPath):
	image = cv2.imread(imagesPath+"/"+file_name)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
	facesEncoding.append(f_coding)
	facesNames.append(file_name.split(".")[0])
#print(facesEncoding)
#print(facesNames)
#cv2.destroyAllWindows()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.1, 5)
    for (x, y, w, h) in faces:
    	face = orig[y:y+h, x:x +w]
    	face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    	actual_face_encoding = face_recognition.face_encodings(image, known_face_locations=[(0, w, h, 0)])[0]
    	result = face_recognition.compare_faces(facesEncoding,actual_face_encoding)
    	print(result)
    	if True in result:
    		index = result.index(True)
    		name = facesNames[index]
    		color= (125,228,0)
    	else:
    		name = "desconocido"
    		color= (50,50,255)
    	cv2.rectangle(frame,(x, y+h), (x+w, y+h+30),color,-1)
    	cv2.rectangle(frame,(x, y), (x+w, y+h),(0,255,0),2)
    	cv2.putText(frame, name, (x,y+h+25),2,2,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('webCam',frame)
    if (cv2.waitKey(1) == ord('s')):
        break
cap.release()
cv2.destroyAllWindows()