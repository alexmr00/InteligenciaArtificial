import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

# Función para cargar imágenes y etiquetas desde una estructura de carpetas
def load_images_labels(base_folder):
    all_images = []
    all_labels = []
    label_names = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]

    for label_index, name in enumerate(label_names):
        person_folder = os.path.join(base_folder, name)
        files = [f for f in os.listdir(person_folder) if os.path.isfile(os.path.join(person_folder, f))]
        for file in files:
            file_path = os.path.join(person_folder, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                all_images.append(image.flatten())
                all_labels.append(label_index)

    return np.array(all_images), np.array(all_labels), label_names

# Función para evaluar el modelo con configuración de hiperparámetros
def evaluate_model(X_train, y_train, X_test, y_test, params):
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"Pérdida: {loss:.4f}")
    return accuracy, loss

# Cargar datos
faces_folder = 'Dataset/Faces'  # Asegúrate de que esta sea la ruta correcta
X, y, label_names = load_images_labels(faces_folder)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración de hiperparámetros
params = {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 200}

# Evaluar el modelo
accuracy, loss = evaluate_model(X_train, y_train, X_test, y_test, params)

# No es necesario graficar aquí a menos que desees visualizar algo específico


