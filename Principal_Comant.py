from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from PIL import ImageTk, Image  
import cv2
import tkinter as tk
import imutils
import numpy as np
# importing os module  
import os
import tkinter.messagebox

def reiniciar_count():
    global count
    count = 0
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
        lblInfo1 = Label(capturar, text="IMAGEN DE ENTRADA:")
        lblInfo1.place(x=5,y=50)
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
            dataPath = 'data'
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
                    tkinter.messagebox.showinfo(title="Alterta",message="Captura Exitosa")
                    reiniciar_count()
                    guardar_face()
                    cap_face_bol()
            #count = count + 1
            
        img = Image.fromarray(frame)
        image = ImageTk.PhotoImage(image=img)
        etip_de_video.configure(image=image)
        etip_de_video.image = image
        etip_de_video.after(10, iniciar)

video2 = None
cap_face = False
def video_Stream_principal():
    global video2
    video2 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    iniciar_principal()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')
dataPath = 'C:/Users/Lyria/Documents/raquel_ia/data' 
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

def iniciar_principal():
    global video2
    ret, frame2 = video2.read()
    frame2 = cv2.flip(frame2,1)
    if ret ==True:
        frame2 = imutils.resize(frame2,width=400, height=300)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        #cv2.COLOR_BGR2GRAY
        auxFrame = frame2.copy()
        auxFrame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(frame2,1.2,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame2, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            assertrostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(frame2,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            if result[1] < 75:
                cv2.rectangle(frame2,(x, y+h), (x+w-50, y+h+30),(0,255,0),-1)
                cv2.rectangle(frame2,(x, y), (x+w, y+h),(0,255,0),2)
                print(imagePaths[result[0]])
                recog_face(imagePaths[result[0]])
                cv2.putText(frame2, '{}'.format(imagePaths[result[0]]), (x,y+h+25),2,0.8,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.rectangle(frame2,(x, y+h), (x+w-50, y+h+30),(0,0,255),-1)
                cv2.rectangle(frame2,(x, y), (x+w, y+h),(0,0,255),2)
                recog_face("desconocido")
                cv2.putText(frame2,'Desconocido',(x,y+h+25),2,0.8,(255,255,255),1,cv2.LINE_AA)
        img = Image.fromarray(frame2)
        image = ImageTk.PhotoImage(image=img)
        etip_video_principal.configure(image=image)
        etip_video_principal.image = image
        etip_video_principal.after(10, iniciar_principal)

def cap_face_bol():
    global cap_face
    print(cap_face)
    imgGuardar = txtNombre.get()
    if is_empty(imgGuardar):
        tkinter.messagebox.showinfo(title="Alerta",message="Nombre Vacio")
    elif image_face==None or is_empty(image_face):
        tkinter.messagebox.showinfo(title="Alerta",message="Imagen Vacia")
    else:
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
def is_empty(txt):
    txt = set(txt)
    return not bool(txt)

def recog_face(jeta):
    x="faces/"+jeta+".jpg"
    image = cv2.imread(x)
    image= imutils.resize(image, height=380)
    imageToShow= imutils.resize(image, width=247)
    imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(imageToShow )
    img = ImageTk.PhotoImage(image=im)
    InputImageRostro.configure(image=img)
    InputImageRostro.image = img

def entrenamiento():
    global dataPath
    #dataPathEntre = 'data'
    peopleList = os.listdir(dataPath)
    print("lista de personas",peopleList)
    labelsEntre = []
    faceDataEntre = []
    labelEntre  = 0
    for nameDir in peopleList:
        personPath = dataPath+'/'+nameDir
        #print("leyendo imagenes")
        for fileName in os.listdir(personPath):
            labelsEntre.append(labelEntre)
            faceDataEntre.append(cv2.imread(personPath+'/'+fileName,0))
            image = cv2.imread(personPath+'/'+fileName,0)
        labelEntre = labelEntre + 1
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("entrenando....")

    face_recognizer.train(faceDataEntre, np.array(labelsEntre))

    face_recognizer.write('C:UsersLyriaDocumentsraquel_ia/modeloLBPHFace.xml')

    #face_recognizer.write('modeloFisherFace.xml')
    print("modelo almanecado")
    tkinter.messagebox.showinfo(title="Alerta",message="Registro Exitoso")


def mostrar_principal():
    capturar.pack_forget()
    principal.pack()
    offVideoSecundario()

def usuario_registro():
    principal.pack_forget()
    capturar.pack()
    offVideoPrincipal()

def onVideoPrincipal():
    video_Stream_principal()

def offVideoPrincipal():
    global video
    video.release()

def onVideoSecundario():
    video_Stream()

def offVideoSecundario():
    global video2
    video2.release()

# Creamos la ventana principal
root = Tk()
root.title("Reconocimiento Facial")
root.geometry("1000x500")
root.resizable(width=True, height=True)

#elemtos menu
menubar = Menu(root)
root.config(menu=menubar)

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="ventana principal",command=mostrar_principal)
filemenu.add_command(label="nuevo usuario",command=usuario_registro)
filemenu.add_separator()
filemenu.add_command(label="Salir", command=root.quit)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Editar Usuario")

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Ayuda")
helpmenu.add_separator()
helpmenu.add_command(label="Acerca de...")

menubar.add_cascade(label="Usuarios", menu=filemenu)
menubar.add_cascade(label="Editar", menu=editmenu)
menubar.add_cascade(label="Ayuda", menu=helpmenu)
##frame de vista principal
principal = Frame(root)
principal.pack()
principal.config(bg="lightblue")
principal.config(width=1000,height=500) 
#principal.pack_forget()
InputImageRostro = Label(principal,bg="black")
InputImageRostro.place(x=500,y=50)
#Etiqueta
etip_video_principal = tk.Label(principal,bg="black")
etip_video_principal.place(x=50,y=50)
botonOnCamara1 = tk.Button(principal, text="enceder camara",cursor = "hand2",width=15,command=onVideoPrincipal, height=2,font=("Calisto MT",12,"bold"))
botonOnCamara1.place(x=150,y=400)
botonOffCamara1 = tk.Button(principal, text="apagar camara",cursor = "hand2",width=15,command=offVideoPrincipal, height=2,font=("Calisto MT",12,"bold"))
botonOffCamara1.place(x=350,y=400)

#frame registro
capturar = Frame(root)
#capturar.pack()
capturar.config(bg="red")
capturar.config(width=1000,height=500)
#boton capturar
boton = tk.Button(capturar, text="iniciar Captura",cursor = "hand2",width=15,command=cap_face_bol, height=2,font=("Calisto MT",12,"bold"))
boton.place(x=480,y=400)
#boton 
# Creamos el botón para elegir la imagen de entrada
btn = Button(capturar, text="Elegir imagen", width=25, command=elegir_imagen)
btn.place(x=5,y=5)
lblInputImage = Label(capturar)
lblInputImage.place(x=5,y=80)
# Label donde se presentará la imagen de salida
lblOutputImage = Label(capturar)
lblOutputImage.place(x=5,y=400)

botonOnCamara2 = tk.Button(capturar, text="enceder camara",cursor = "hand2",width=15,command=onVideoSecundario, height=2,font=("Calisto MT",12,"bold"))
botonOnCamara2.place(x=100,y=400)
botonOffCamara2 = tk.Button(capturar, text="apagar camara",cursor = "hand2",width=15,command=offVideoSecundario, height=2,font=("Calisto MT",12,"bold"))
botonOffCamara2.place(x=300,y=400)
#Etiqueta
botonEntrenar = tk.Button(capturar, text="registrar rostro",cursor = "hand2",width=15,command=entrenamiento, height=2,font=("Calisto MT",12,"bold"))
botonEntrenar.place(x=660,y=400)

labelNombre = tk.Label(capturar, text = "Nombre")
labelNombre.place(x=5,y=310)
txtNombre = tk.Entry(capturar, width=20)
txtNombre.place(x=5,y=340)
   
etip_de_video = tk.Label(capturar,bg="black")
etip_de_video.place(x=220,y=50)

#hide_captura()
video_Stream_principal()
#video_Stream()
#inicio_frame()
root.mainloop()