import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt

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

# Cargar datos
faces_folder = 'Dataset/Faces'  # Asegúrate de que esta sea la ruta correcta
X, y, label_names = load_images_labels(faces_folder)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración de GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
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

# Graficar resultados de GridSearchCV
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

plt.figure(figsize=(10, 5))
plt.plot(range(len(mean_test_scores)), mean_test_scores, marker='o', linestyle='-')
plt.title('GridSearchCV Testing Scores')
plt.xlabel('Config Index')
plt.ylabel('Mean CV Score')
plt.grid(True)
plt.show()
