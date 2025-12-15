from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib as plt
from torchvision.transforms import ToPILImage

"""
Esta clase contiene 3 modelos uno para cada atributo y entrena los 3 modelos a la vez para optimizar el tiempo
Clase Agente contiene la siguiente info:
- Dataset
- Modelo de la red que se quiera entrenar
- Numero de epoch
- Criterio y optimizador

Ademas contiene los metodos:

- entrenamientoTest = divide el dataset en entrenamiento y test
- entrenarModelo : entrena el modelo
"""


class Agente:


    def __init__(self,modeloEdad,modeloGenero, modeloRaza, criterioEdad, criterioGenero, criterioRaza, device,lr):

        #Datos principales
        self.device = device #CPU o GPU depende del ordenador
        self.lr = lr

        #Modelo edad
        self.modeloEdad= modeloEdad.to(device) #Red neuronal para la edad
        self.criterioEdad = criterioEdad
        self.optimizadorEdad = optim.Adam(self.modeloEdad.parameters(), lr=self.lr )

        #Modelo  genero
        self.modeloGenero= modeloGenero.to(device) #Red neuronal para la edad
        self.criterioGenero = criterioGenero
        self.optimizadorGenero = optim.Adam(self.modeloGenero.parameters(), lr=self.lr )

        #Modelo raza
        self.modeloRaza= modeloRaza.to(device) #Red neuronal para la edad
        self.criterioRaza = criterioRaza
        self.optimizadorRaza = optim.Adam(self.modeloRaza.parameters(), lr=self.lr )
        

    def entrenarModelo(self, train_data,test_data,tarea, num_epochs):

        """
        Funcion que entrena el modelo segun la tarea especifica: Edad, Genero o Raza
        Además calcula el error (loss) y la tasa de aciertos(accuracy) con dos conjuntos:
            - Conjunto de entrenamiento
            - Conjunto test mediante la funcion validacion (con este conjunto NO entrena; solo predice resultados)
        Para que no ralentize el entrenamiento la validadcion se realiza cada 5 epochs
        Esto será util para hacer gráficas y ver si el modelo esta sobreaprendiendo o no
        """

        #Mapeamos el modelo, optimizador y criterio segun la tarea
        modelo = getattr(self, f'modelo{tarea.capitalize()}')
        optimizador = getattr(self, f'optimizador{tarea.capitalize()}')
        criterio = getattr(self, f'criterio{tarea.capitalize()}')

        # Selector de etiquetas según la tarea
        idx_etiqueta = {"Edad": 1, "Genero": 2, "Raza": 3}[tarea.capitalize()]

        train_loss = []
        train_accuracy = []

        test_loss = []
        test_accuracy = []

        #Comienza entrenamiento
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_accuracy  = 0.0

            for batch in train_data:
                imagenes = batch[0].to(self.device)
                etiquetas = batch[idx_etiqueta].to(self.device)

                optimizador.zero_grad()
                predicciones = modelo(imagenes)

                if tarea.lower() == "edad":
                    predicciones = predicciones.squeeze()

                loss = criterio(predicciones, etiquetas)
                loss.backward()
                optimizador.step()

                #Datos evolucion entrenamiento
                running_loss += loss.item()
                
                if tarea.lower() != "edad": 
                    _, predicciones = torch.max(predicciones, 1)
                else:
                    predicciones = predicciones.int()
                    
                running_accuracy += (predicciones == etiquetas).sum().item()
            
            #Fin de epoch: Mostramos el avance
            running_loss = running_loss / (len(train_data)*train_data.batch_size)
            running_accuracy = running_accuracy / (len(train_data)*train_data.batch_size)
    
            print(f'[{epoch + 1}, {tarea}] loss: {running_loss :.3f}')
            print(f'[{epoch + 1}, {tarea}] accuracy: {running_accuracy:.3f}')

            #Actualizamos informacion
            train_loss.append(running_loss)
            train_accuracy.append(running_accuracy)
            #Validacion
            if epoch % 5 == 0 or epoch == num_epochs-1: #Cada 5 epochs vemos como se comporta el modelo con el conjunto de test
                running_test_loss, running_test_accuracy = self.validarModelo(test_data,tarea)
                test_loss.append(running_test_loss)
                test_accuracy.append(running_test_accuracy)
        

        print(f"Finished Training {tarea}")
        return train_loss, train_accuracy,test_loss,test_accuracy


    def validarModelo(self, test_data,tarea):
        
        """
        Funcion que se encarga de validar el conjunto de test, es decir, ver como se comporta la red neuronal en determinado
        momento calculando el error (loss) y la tasa de aciertos (accuracy)
        """

        modelo = getattr(self, f'modelo{tarea.capitalize()}')
        criterio = getattr(self, f'criterio{tarea.capitalize()}')

        # Selector de etiquetas según la tarea
        idx_etiqueta = {"Edad": 1, "Genero": 2, "Raza": 3}[tarea.capitalize()]

        
        running_loss = 0.0
        running_accuracy = 0.0

        with torch.no_grad():  #Con esto nos aseguramos de que NO entrene el modelo
            
            for batch in test_data:
                
                #Extraemos los datos del batch
                imagenes = batch[0].to(self.device)
                etiquetas = batch[idx_etiqueta].to(self.device)

                #Pasamos las imagenes por el modelo
                predicciones = modelo(imagenes)

                if tarea.lower() == "edad":
                    predicciones = predicciones.squeeze()
    

                loss = criterio(predicciones, etiquetas)

                #Datos evolucion entrenamiento
                running_loss += loss.item()
                
                if tarea.lower() != "edad": 
                    _, predicciones = torch.max(predicciones, 1)
                else:
                    predicciones = predicciones.int()
                    
                running_accuracy += (predicciones == etiquetas).sum().item()


            running_loss = running_loss / (len(test_data)*test_data.batch_size)
            running_accuracy = running_accuracy / (len(test_data)*test_data.batch_size)
        
        return running_loss,running_accuracy



    def resultados(self,datos):

        """
        Esta funcion, dado un conjunto de datos, recopila las etiqueta reales Genero, Edad y  Raza (generosTotal, edadesTotal,razasTotal)
        y las etiquetas que predecin los respectivos nmodelos (generosPredTotal,edadesPredTotal,razasPredTotal)
        De esta forma luego será mas facil, usando las libreria de SckitLearn, calcular métricas resultantes 
        """
        generosTotal = torch.tensor([]).to(self.device)
        generosPredTotal = torch.tensor([]).to(self.device)

        edadesTotal = torch.tensor([]).to(self.device)
        edadesPredTotal = torch.tensor([]).to(self.device)

        razasTotal = torch.tensor([]).to(self.device)
        razasPredTotal = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for batch in datos:

                imagenes,edades,generos,razas = batch
                imagenes, edades, generos,razas = imagenes.to(self.device),edades.to(self.device),generos.to(self.device),razas.to(self.device)

                #Genero
                generosPred = self.modeloGenero(imagenes)
                _, generosPred = torch.max(generosPred, 1)

                generosPredTotal =torch.cat((generosPredTotal,generosPred),dim = 0)
                generosTotal = torch.cat((generosTotal,generos),dim = 0)

                #Edad
                edadesPred = self.modeloEdad(imagenes)
                edadesPredTotal = torch.cat((edadesPredTotal,edadesPred.squeeze().int()),dim = 0)
                edadesTotal = torch.cat((edadesTotal,edades), dim = 0)


                #Raza
                razasPred = self.modeloRaza(imagenes)
                _, razasPred = torch.max(razasPred, 1)
        
                razasPredTotal =torch.cat((razasPredTotal,razasPred),dim = 0)
                razasTotal = torch.cat((razasTotal,razas),dim = 0)

        return edadesTotal.to("cpu"),edadesPredTotal.to("cpu"),generosTotal.to("cpu"),generosPredTotal.to("cpu"),razasTotal.to("cpu"),razasPredTotal.to("cpu")
    



