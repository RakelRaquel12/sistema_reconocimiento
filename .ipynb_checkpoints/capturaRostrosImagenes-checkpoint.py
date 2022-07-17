import cv2
import os
import imutils


personName = 'Raquel'
#personName = 'Josefina'
dataPath = 'C:/Users/Lyria/Documents/raquel_ia2/data'
personPath = dataPath+"/"+personName
if not os.path.exists(personPath):
	os.makedirs(personPath)

imagesPath = "C:/Users/Lyria/Documents/raquel_ia2/db/Raquel"
#imagesPath = "C:/Users/Lyria/Documents/raquel_ia2/db/Josefina"
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

for imageName in os.listdir(imagesPath):
	print(imageName)
	image  = cv2.imread(imagesPath+"/"+imageName,0)
	#comenta esta linea cuando son fotos pequenias
	#para que el rostro se capture de mejor manera
	image = cv2.resize(image,(750,1000))
	faces = faceClassif.detectMultiScale(image, 1.1, 5)
	for (x, y, w, h) in faces:
		face = image[y:y+h, x:x+w]
		face = cv2.resize(face,(150,150))
		cv2.imwrite(personPath+"/rostro_{}.jpg".format(count),face)
		count+=1