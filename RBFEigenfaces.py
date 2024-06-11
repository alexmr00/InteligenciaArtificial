import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from skimage import color
import matplotlib.pyplot as plt
from skimage.feature import hog

def load_images_and_labels(base_folder):
    images = []
    labels = []
    label_names = [name for name in os.listdir(base_folder) if not name.startswith('.')]
    label_map = {name: idx for idx, name in enumerate(label_names)}

    for label in label_names:
        person_dir = os.path.join(base_folder, label)
        if os.path.isdir(person_dir):
            for image_filename in os.listdir(person_dir):
                if not image_filename.startswith('.'):
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = color.rgb2gray(image)
                        image = cv2.resize(image, (64, 64))
                        images.append(image.flatten())
                        labels.append(label_map[label])
    return np.array(images), np.array(labels), label_names

# Cargar imágenes
dataset_path = 'Dataset/Faces'
images, labels, label_names = load_images_and_labels(dataset_path)

# Aplicar PCA para extracción de características (Eigenfaces)
pca = PCA(n_components=150, whiten=True, random_state=42)
images_pca = pca.fit_transform(images)  # Asegúrate de que 'images' ya es un array de NumPy

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(images_pca, labels, test_size=0.2, random_state=42)

# Entrenar el modelo SVM con kernel RBF
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)

# Mostrar la precisión y la pérdida logarítmica
print(f'Precisión del modelo: {accuracy:.4f}')
print(f'Pérdida logarítmica (Log Loss): {logloss:.4f}')
