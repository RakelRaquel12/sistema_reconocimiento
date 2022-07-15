import cv2
import os
import numpy as np
personas  = ["raquel"]
def dameCaras(img):
  gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
  caras = face_cascade.detectMultiScale(gris,scaleFactor=1.2,minNeighbors=5);
  if (len(caras)==0):
    return None, None
  (x,y,w,h) = caras[0]
  return gris [y:y+w, x:x+h],caras[0]
def prepararDatosEntrenamiento(ruta):
  directorios = os.listdir(ruta)
  caras = []
  labels = []
  for nombreDire in directorios:
    label = int (nombreDire)
    rutaDirectorioPersona = ruta+ "/" + nombreDire
    listaImagenesPersona = os.listdir(rutaDirectorioPersona)
    for nombreImagen in listaImagenesPersona:
      rutaImagen = rutaDirectorioPersona + "/" + nombreImagen
      imagen  = cv2.imread(rutaImagen)
      scale_percent = 10
      width = int(imagen.shape[1] * scale_percent / 100)
      height = int(imagen.shape[0] * scale_percent / 100)
      dsize = (width, height)
      output = cv2.resize(imagen, dsize)
      cv2.imshow("entrenando........", output)
      cv2.waitKey(10)
      face, rect = dameCaras(imagen)
      if face is not None:
        caras.append(face)
        labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
  return caras, labels


def pintarRectangulo(img, rect):
  (x, y, w, h) = rect
  cv2.rectangle(img,(x, y), (x+w, y+h), (0,255,0),2)

def escribirTexto(img, text, x, y):
  cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predecir(imagenTest):
  img = imagenTest.copy()
  cara, rect = dameCaras(img)
  label = recognizer.predict(cara)
  if label is not None:
    label_text = personas[label[0]]
    pintarRectangulo(img. rect)
    escribirTexto(img, label_text, rect[0],rect[1]-5)
  return img
print("preparando datos...")
caras, labels = prepararDatosEntrenamiento("Fotos")
print("datos preparados")
print("total caras: ",len(caras))
print("total labels: ",len(labels))
recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.train(caras,np.array(labels))

testImg0 = cv2.imread("test/foto_1.jpeg",0)
cv2.imshow("lectura........", testImg0)
#testImg1 = cv2.imread("test/foto_2")
#testImg2 = cv2.imread("test/foto_3")
cv2.waitKey(100)
predictImg0 = predecir(testImg0)
#predictImg1 = predecir(testImg1)
#predictImg2 = predecir(testImg2)
#print("prediccion completa")
cv2.imshow(personas[0],predictImg0)
#cv2.imshow(personas[1],predictImg1)
#cv2.imshow(personas[2],predictImg2)
cv2.waitkey(0)
cv2.destroyAllWindows()