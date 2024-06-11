import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from skimage.feature import hog
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

# Entrenar el modelo SVM con kernel RBF
model = SVC(kernel='rbf', probability=True)  
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Probabilidades para calcular la log loss
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)  # Calcular log loss

# Mostrar la precisión y la pérdida logarítmica
print(f'Precisión del modelo: {accuracy:.4f}')
print(f'Pérdida logarítmica (Log Loss): {logloss:.4f}')




