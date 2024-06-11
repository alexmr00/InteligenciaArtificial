import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
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
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
            if image is not None:
                image = cv2.resize(image, (64, 64))  # Redimensionar para tener consistencia
                all_images.append(image.flatten())
                all_labels.append(label_index)

    return np.array(all_images), np.array(all_labels), label_names

# Aplicar PCA para extracción de características (Eigenfaces)
def apply_pca(X, n_components=150):
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Función para evaluar el modelo con diferentes números de estimadores
def evaluate_estimators(estimator_range, X_train, y_train, X_test, y_test):
    accuracies = []
    losses = []
    
    for n_estimators in estimator_range:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        losses.append(log_loss(y_test, y_prob))
    
    # Imprimir precisión y pérdida finales
    print(f"Precisión final: {accuracies[-1]:.4f}")
    print(f"Pérdida final: {losses[-1]:.4f}")
    
    return accuracies, losses

# Cargar datos
faces_folder = 'Dataset/Faces'  
X, y, label_names = load_images_labels(faces_folder)

# Aplicar PCA
X_pca, pca = apply_pca(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Rango de estimadores a evaluar
estimator_range = range(10, 110, 30)

# Evaluar el modelo
accuracies, losses = evaluate_estimators(estimator_range, X_train, y_train, X_test, y_test)

# Graficar resultados
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(estimator_range, accuracies, marker='o', linestyle='-', color='b')
plt.title('Precisión del modelo vs. Número de estimadores')
plt.xlabel('Número de estimadores')
plt.ylabel('Precisión')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(estimator_range, losses, marker='o', linestyle='-', color='r')
plt.title('Pérdida logarítmica vs. Número de estimadores')
plt.xlabel('Número de estimadores')
plt.ylabel('Pérdida logarítmica')
plt.grid(True)

plt.tight_layout()
plt.show()
