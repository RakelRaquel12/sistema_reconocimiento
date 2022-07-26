import cv2
import os
import numpy as np

dataPath = 'G:\auxiliar investigacion\sistema\Nueva carpeta\sistema_\data'
peopleList = os.listdir(dataPath)
print("lista de personas",peopleList)

labels = []
faceData = []
label  = 0

for nameDir in peopleList:
	personPath = dataPath+'/'+nameDir
	print("leyendo imagenes")
	for fileName in os.listdir(personPath):
		print('Rostros: ',nameDir+'/'+fileName)
		labels.append(label)
		faceData.append(cv2.imread(personPath+'/'+fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('imagen',image)
		#cv2.waitKey(10)
	label = label + 1
#cv2.destroyAllWindows()
#print ('labels = ',labels)
print ('Numero de etiqueta 0: ',np.count_nonzero(np.array(labels)==0))
print ('Numero de etiqueta 1: ',np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
print("entrenando....")

face_recognizer.train(faceData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')

#face_recognizer.write('modeloFisherFace.xml')
print("modelo almanecado")