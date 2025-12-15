#Vamos a trabajar con pytorch y este permite crear tu propio dataset
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

"""
Esta clase sirve para tratar las imagenes de manera mas optima con la biblioteca pytorch
Es una clase que hereda de Dataset (perteneciente a pytorch)

Estructura de las imagenes:
- 200x200 pixeles
- Nombre imagen : edad_genero_raza_fecha.jpg
- edad: cualquier numero entre 1 y 100
- genero : 0 = hombre y 1= mujer
- raza : 5 clases

__getitem__ se encarga de leer el nombre de la foto y separar los datos; los convierte en tensores y si hay
algun transformador de los datos los transforma
Devuelve la imagen, el año, el genero y la raza

"""

class Dataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir) #lista de los nombres de las fotos
        self.img_dir = img_dir #Ruta principal donde estan las imagenes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        nombre_imagen = os.path.join(self.img_dir, self.img_labels[idx])  #Ruta completa
        imagen =  Image.open(nombre_imagen) #abrimos la imagen

        # Extraer la info del nombre del archivo
        parts = self.img_labels[idx].split('_')  # Separar por guiones bajos

        age = torch.tensor(int(parts[0])).float()  # Primer elemento: la edad
        gender = torch.tensor(int(parts[1]))  # Segundo elemento: el género
        race = torch.tensor(int(parts[2]))
        
        if self.transform:

          imagen = self.transform(imagen)  # Aplicar transformaciones a la imagen

        if self.target_transform:
          age = torch.tensor(int(self.target_transform(age)))  # Aplicar transformaciones a la edad (opcional)

        return imagen, age,gender,race
    
def entrenamientoTest(dataset,test_size = 0.2,batch_size =4):
    """
    entrenamientoTest se encarga de dividir el dataset total en conjunto de entrenamiento y test
    """
    #Primero realizamos una division de  indices
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    #Dividimos los indices
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # Crear los DataLoaders con los indices que hemos repartidos
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
  
    #Dividimos ambos conjuntos en batches
    return train_dataloader,test_dataloader


def cargarDatos(dataloader, num_imagenes=500, categoria="raza"):
    """
    cargarDatos carga las imagenes y sus correspondiente etiqueta en arrays
    Esta funcion se utiliza para el modelo KNN, pues requiere que previamente los datos esten
    almacenados en una lista o array

    Args:
    -dataloader: objeto que permite acceder a las imagenes y sus correspondientes etiquetas
    -num imagenes: numero de imagenes que se desea cargar del conjunto dataloader
    - categoria: etiqueta que se desea almacenar 
    Extrae un subconjunto de imágenes y sus etiquetas de una categoría específica.
    """
    X = []  # Para almacenar las características (imágenes aplanadas)
    Y= []  # Para almacenar las etiquetas correspondientes a la categoría

    # Iteramos sobre los mini-lotes en el DataLoader
    for imagenes, edades, generos, razas in dataloader:
        if len(X) >= num_imagenes:  # Si ya tenemos suficientes imágenes, salimos
            break

        # Aplanamos las imágenes de tamaño (batch_size, 3, 200, 200) a (batch_size, 120000)
        imagenes_flat = imagenes.view(imagenes.size(0), -1).cpu().numpy()

        # Seleccionamos las etiquetas según la categoría
        if categoria == "raza":
            etiquetas = razas.cpu().numpy()
        elif categoria == "edad":
            etiquetas = edades.cpu().numpy()
        elif categoria == "genero":
            etiquetas = generos.cpu().numpy()

        # Añadir las imágenes y las etiquetas al subconjunto
        X.append(imagenes_flat)
        Y.append(etiquetas)

    # Concatenamos todas las imágenes y las etiquetas
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y



def clasificarEdad(edad):
    return edad//10




def graficaEntrenamiento(train_loss,train_accuracy,test_loss,test_accuracy,epochs,nombre):
    x1 = np.arange(0,epochs)
    x2 = np.arange(0,epochs+1,5)

    
    fig,axs = plt.subplots(1,2)

    #Loss
    axs[0].plot(x1,train_loss,label = "Train Loss", color = 'b')
    axs[0].plot(x1,test_loss,label = "Test Loss", color = 'r')
    axs[0].set_title('Loss')  # Título del primer gráfico
    axs[0].set_xlabel('Epochs')  # Etiqueta del eje X
    axs[0].set_ylabel('Loss')  # Etiqueta del eje Y
    axs[0].legend()
    axs[0].grid(True)

    #Train
    axs[1].plot(x1,train_accuracy,label = "Train Accuracy", color = 'b')
    axs[1].plot(x1,test_accuracy,label = "Test Accuracy", color = 'r')
    axs[1].set_title('Accuracy')  # Título del primer gráfico
    axs[1].set_xlabel('Epochs')  # Etiqueta del eje X
    axs[1].set_ylabel('Acurracy')  # Etiqueta del eje Y
    axs[1].legend()
    axs[1].grid(True)
    
    plt.grid(True)
    plt.tight_layout()  # Ajusta el espaciado entre los subgráficos
    plt.savefig(f'.//Graficas//{nombre}')


def mostrarImg(img):
    """
    funcion que muestra las imagenes
    """ 
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def convertirRedimensionar(imagen_path, output_path):
    """
    funcion para hacer pruebas con nuestras fotos
    """
    # Abrir la imagen
    with Image.open(imagen_path) as img:
        # Si la imagen tiene un canal alfa (transparencia), convertir a RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Redimensionar la imagen a 200x200
        img_resized = img.resize((200, 200))
        
        # Guardar la imagen en formato JPG
        img_resized.save(output_path, 'JPEG')









def guardar_modelo(modelo, nombre_archivo):
    joblib.dump(modelo, nombre_archivo)
    print(f"Modelo guardado en {nombre_archivo}")

def recuperar_modelo(nombre_archivo):
    try:
        # Cargar el modelo desde el archivo
        modelo = joblib.load(nombre_archivo)
        print(f"Modelo cargado desde {nombre_archivo}")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo desde {nombre_archivo}: {e}")
        return None






