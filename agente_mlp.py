from sklearn.metrics import *
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.init as init
import importlib
import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
import agente_cnn
import agente_knn
from torch.utils.data import Subset
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def train_and_evaluate(model, train_loader, test_loader, etiqueta, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()  # Usamos CrossEntropyLoss para clasificación
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Diccionario para mapear las etiquetas
    etiqueta_indices = {'edad': 0, 'genero': 1, 'raza': 2}
    idx_etiqueta = etiqueta_indices[etiqueta]  # Selección de etiqueta

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Establecemos el modelo en modo de entrenamiento
        running_loss = 0.0
        correct = 0
        total = 0

        # Entrenamiento
        for data in train_loader:
            images = data[0].to(device)  # Obtener las imágenes
            labels = data[1:]  # Obtener las etiquetas (todas las etiquetas devueltas por ImgDataset)
            target_labels = labels[idx_etiqueta].to(device)  # Etiqueta seleccionada (por ejemplo, 'genero')

            optimizer.zero_grad()  # Resetear los gradientes
            outputs = model(images)  # Pasar las imágenes por el modelo

            loss = criterion(outputs, target_labels.long())  # Calcular la pérdida
            loss.backward()  # Retropropagar el error
            optimizer.step()  # Actualizar los parámetros

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # Obtener la clase predicha
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total  # Precisión de entrenamiento

        # Evaluación en test set
        model.eval()  # Establecemos el modelo en modo de evaluación
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():  # No necesitamos gradientes en la evaluación
            for data in test_loader:
                images = data[0].to(device)
                labels = data[1:]
                target_labels = labels[idx_etiqueta].to(device)

                outputs = model(images)

                loss = criterion(outputs, target_labels.long())
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += target_labels.size(0)
                correct_test += (predicted == target_labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test / total_test  # Precisión de test

        # Guardamos las métricas
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluar(modelo, dataloader, etiqueta="genero", dispositivo="cpu"):
    """
    Evalúa un modelo MLP de PyTorch usando un DataLoader.
    Calcula precisión (accuracy), RMSE y muestra precision/recall/F1-score.

    Args:
        modelo: Modelo MLP ya entrenado (PyTorch).
        dataloader: DataLoader que entrega (imagenes, edades, generos, razas).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').
        dispositivo: 'cpu' o 'cuda' dependiendo de dónde está el modelo.

    Returns:
        accuracy: Precisión del modelo en el conjunto de prueba.
        rmse: Raíz del error cuadrático medio.
        report: Reporte de clasificación.
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
    """
    modelo.eval()
    modelo.to(dispositivo)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imagenes, edades, generos, razas in dataloader:
            imagenes = imagenes.to(dispositivo)
            entradas = imagenes.view(imagenes.size(0), -1)

            if etiqueta == "raza":
                etiquetas = razas.to(dispositivo)
            elif etiqueta == "edad":
                etiquetas = edades.to(dispositivo)
            elif etiqueta == "genero":
                etiquetas = generos.to(dispositivo)
            else:
                raise ValueError("Etiqueta debe ser 'raza', 'edad' o 'genero'")

            salidas = modelo(entradas)

            if salidas.shape[1] > 1:  # Clasificación multiclase
                predicciones = torch.argmax(salidas, dim=1)
            else:  # Regresión o binaria
                predicciones = salidas.round().squeeze()

            y_true.extend(etiquetas.cpu().numpy())
            y_pred.extend(predicciones.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return  y_true, y_pred




def predict_fn_mlp(images_np, modelo, device):
    """
    Función que adapta el modelo MLP de PyTorch para LIME.
    Toma imágenes en formato numpy (H, W, C), las convierte a tensores y aplica el modelo.
    
    Devuelve: numpy array con probabilidades (batch_size, num_classes)
    """
    modelo.eval()
    images = torch.tensor(images_np.transpose((0, 3, 1, 2)), dtype=torch.float32)  # (N, H, W, C) → (N, C, H, W)
    images = images.to(device)
    images = images.reshape(images.size(0), -1)  # <-- Cambio aquí
    with torch.no_grad():
        logits = modelo(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def lime_explain_and_visualize_mlp(modelo, imagen, device='cpu', num_samples=1000):
    """
    Genera una explicación de LIME para una imagen usando un MLP de PyTorch y visualiza el resultado.
    
    Args:
        modelo: Modelo MLP entrenado (PyTorch).
        imagen: Imagen de entrada (tensor de tamaño (C, H, W)).
        device: 'cpu' o 'cuda'.
        num_samples: Número de muestras para LIME.
    """
    imagen_np = imagen.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=imagen_np.astype(np.double),
        classifier_fn=lambda x: predict_fn_mlp(x, modelo, device),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )

    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(imagen_np)
    ax[0].set_title("Imagen Original")
    ax[0].axis('off')

    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title("Explicación LIME")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

import torch

def predecir_instancia_mlp(modelo, imagen, etiqueta="genero"):
    """
    Evalúa un modelo MLP de PyTorch usando una única instancia de imagen y devuelve la predicción.

    Args:
        modelo: Modelo MLP de PyTorch ya entrenado.
        imagen: Instancia de imagen a evaluar (tensor de tamaño (C, H, W)).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').

    Returns:
        prediccion: La predicción de la clase para la instancia dada.
    """
    # Asegurarse de que la imagen esté en el formato adecuado (batch_size, C, H, W)
    imagen = imagen.unsqueeze(0)  # Añadir dimensión de batch (1, C, H, W)
    
    # Pasar la imagen por la red (asegurarse de que esté en modo de evaluación)
    modelo.eval()
    
    with torch.no_grad():
        # Predecir usando el modelo
        prediccion = modelo(imagen)  # Esto devuelve logits
        _, prediccion_clase = torch.max(prediccion, 1)  # Obtener la clase con la mayor probabilidad
        
    return prediccion_clase.item()  # Regresar la clase predicha como un valor escalar

def predecir_varias_instancias_mlp(modelo, imagenes, etiqueta="genero"):
    """
    Evalúa un modelo MLP de PyTorch usando un batch de imágenes y devuelve las predicciones para cada instancia.

    Args:
        modelo: Modelo MLP de PyTorch ya entrenado.
        imagenes: Batch de imágenes a evaluar (tensor de tamaño (N, C, H, W), donde N es el tamaño del batch).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').

    Returns:
        predicciones: Lista de clases predichas para cada imagen en el batch.
    """
    # Asegurarse de que las imágenes estén en el formato adecuado (batch_size, C, H, W)
    # Si las imágenes ya están en el formato correcto, no es necesario modificar nada
    
    # Pasar las imágenes por el modelo (asegurarse de que esté en modo de evaluación)
    modelo.eval()

    with torch.no_grad():
        # Predecir usando el modelo (esto devuelve logits para cada imagen en el batch)
        logits = modelo(imagenes)
        
        # Aplicar softmax para convertir los logits en probabilidades
        probabilidades = F.softmax(logits, dim=1)
        
        # Obtener la clase con la mayor probabilidad para cada imagen
        _, predicciones_clase = torch.max(probabilidades, 1)
        
    return predicciones_clase.cpu().numpy()  # Devolver las clases predichas como un array de numpy
