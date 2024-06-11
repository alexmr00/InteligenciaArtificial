import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from skimage import color
import matplotlib.pyplot as plt

def load_images_and_labels(base_folder):
    images = []
    labels = []
    label_names = [name for name in os.listdir(base_folder) if not name.startswith('.')]
    label_map = {name: idx for idx, name in enumerate(label_names)}

    for label in label_names:
        person_dir = os.path.join(base_folder, label)
        if os.path.isdir(person_dir):  # Asegúrate de que sea un directorio
            for image_filename in os.listdir(person_dir):
                if not image_filename.startswith('.'):  # Ignorar archivos ocultos
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Convertir la imagen a escala de grises
                        image = color.rgb2gray(image)
                        image = cv2.resize(image, (64, 64))  # Redimensionar para uniformidad
                        # Añadir la imagen procesada al array de imágenes
                        images.append(image.flatten())  # Aplanar la imagen para hacerla compatible con algoritmos de ML
                        labels.append(label_map[label])
    return np.array(images), np.array(labels)

# Cargar imágenes
dataset_path = 'Dataset/Faces'  # Asegúrate de especificar la ruta correcta
images, labels = load_images_and_labels(dataset_path)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Parámetros para GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Parámetro de regularización
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # Coeficiente del kernel
    'kernel': ['rbf']  # Tipo de kernel
}

# Crear el modelo y GridSearchCV
model = SVC(probability=True)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Mejores parámetros y modelo
print("Mejores parámetros encontrados:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predecir en el conjunto de prueba
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)

# Mostrar la precisión y la pérdida logarítmica
print(f'Precisión del modelo: {accuracy:.4f}')
print(f'Pérdida logarítmica (Log Loss): {logloss:.4f}')
