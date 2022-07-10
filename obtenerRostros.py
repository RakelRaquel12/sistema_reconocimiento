import cv2
import os

imagesPath = "C:/Users/Lyria/Documents/raquel_ia/Fotos/0"

if not os.path.exists("faces"):
	os.makedirs("faces")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

for imageName in os.listdir(imagesPath):
	print(imageName)
	image  = cv2.imread(imagesPath+"/"+imageName)
	image = cv2.resize(image,(750,1000))
	faces = faceClassif.detectMultiScale(image, 1.1, 5)
	for (x, y, w, h) in faces:
		face = image[y:y+h, x:x+w]
		face = cv2.resize(face,(150,150))
		cv2.imwrite("faces/"+str(count)+".jpg",face)
		count+=1