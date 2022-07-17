from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import tkinter as tk
import imutils
import numpy as np
# importing os module  
import os
image_face = None
def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    if len(path_image) > 0:
        global image
        #path
        global image_face
        image_face = path_image
        print(image_face)
        # Leer la imagen de entrada y la redimensionamos
        image = cv2.imread(path_image)
        image= imutils.resize(image, height=380)
        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)
        lblInputImage.configure(image=img)
        lblInputImage.image = img
        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)
        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""
video = None
def video_Stream():
	global video
	video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	iniciar()
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0
def iniciar():
    global video
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    if ret ==True:
        frame = imutils.resize(frame,width=400, height=300)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.COLOR_BGR2GRAY
        auxFrame = frame.copy()
        auxFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(frame,1.2,5)
        #print(cap_face)
        if cap_face==True:
            #count = 0
            personName = txtNombre.get()
            #print(personName)
            #print(count)
            dataPath = 'Data'
            personPath = dataPath+"/"+personName
            if not os.path.exists(personPath):
                os.makedirs(personPath)
            for (x,y,w,h) in faces:
                global count
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
                count = count + 1
                #cv2.imshow('frame',frame)
                #k =  cv2.waitKey(1)
                print(count)
                if count >= 50:
                    reiniciar_count()
                    guardar_face()
                    cap_face_bol()
            #count = count + 1
            
        img = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=img)
        etip_de_video.configure(image=image)
        etip_de_video.image = image
        etip_de_video.after(10, iniciar)
def cap_face_bol():
    global cap_face
    print(cap_face)
    if cap_face:
        cap_face=False
    else:
        cap_face=True
def capturar_rostros():
	personName = txtNombre.get()
	print(personName)
	dataPath = 'data'
	personPath = dataPath+"/"+personName
	if not os.path.exists(personPath):
		os.makedirs(personPath)
def reiniciar_count():
    global count
    count = 0
def guardar_face():
    global image_face
    image_path = image_face
    img = cv2.imread(image_path)
    dataPath = 'faces'
    os.chdir(dataPath)
    filename = txtNombre.get()+".jpg"
    cv2.imwrite(filename, img)
# Creamos la ventana principal
root = Tk()
root.title("Reconocimiento Facial")
root.geometry("700x500")
# Label donde se presentará la imagen de entrada
lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)
# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=1, rowspan=6)
#colores
fondo_boton = "#5e17eb"
#botones
#boton = tk.Button(root, text="iniciar video",bg=fondo_boton,relief = "flat",cursor = "hand2",command=video_Stream,width=15, height=2,font=("Calisto MT",12,"bold"))
cap_face = False
#boton.place(x=180,y=400)
boton = tk.Button(root, text="iniciar Captura",cursor = "hand2",width=15,command=cap_face_bol, height=2,font=("Calisto MT",12,"bold"))

boton.place(x=180,y=400)
#Etiqueta
etip_de_video = tk.Label(root,bg="black")
etip_de_video.place(x=220,y=50)
# Creamos los radio buttons y la ubicación que estos ocuparán

labelNombre = tk.Label(root,
                    text = "Nombre")
labelNombre.grid(column=0, row=6, ipadx=5, pady=5)

txtNombre = tk.Entry(root, width=20)
txtNombre.grid(column=0, row=7, padx=10, pady=5, sticky=tk.N)
# Creamos el botón para elegir la imagen de entrada
btn = Button(root, text="Elegir imagen", width=25, command=elegir_imagen)
btn.grid(column=0, row=0, padx=5, pady=5)
video_Stream()
root.mainloop()