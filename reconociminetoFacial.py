import cv2
import os
dataPath = 'C:/Users/Lyria/Documents/raquel_ia2/data' 
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('raquel2.mp4')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        #cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        # LBPHFace
        if result[1] < 75:
            cv2.rectangle(frame,(x, y+h), (x+w-50, y+h+30),(0,255,0),-1)
            cv2.rectangle(frame,(x, y), (x+w, y+h),(0,255,0),2)
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x,y+h+25),2,0.8,(255,255,255),1,cv2.LINE_AA)
        else:
            cv2.rectangle(frame,(x, y+h), (x+w-50, y+h+30),(0,0,255),-1)
            cv2.rectangle(frame,(x, y), (x+w, y+h),(0,0,255),2)
            cv2.putText(frame,'Desconocido',(x,y+h+25),2,0.8,(255,255,255),1,cv2.LINE_AA)
            
    if (cv2.waitKey(1) == ord('s')):
        break
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()