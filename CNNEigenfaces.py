import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras import layers, models, Input
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# Función para cargar imágenes y etiquetas
def load_images_and_labels(base_path):
    images = []
    labels = []
    label_names = os.listdir(base_path)

    for label_name in label_names:
        person_path = os.path.join(base_path, label_name)
        # Asegúrate de que person_path es un directorio antes de proceder
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                # Procesa sólo archivos de imagen, asegurando que no sea un directorio
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
                        image = cv2.resize(image, (32, 32))  # Redimensionar a 32x32
                        images.append(image)
                        labels.append(label_name)
    return np.array(images), np.array(labels), label_names


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        y = np.squeeze(y)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    return categorical


# Cargar y preparar datos
images, labels, label_names = load_images_and_labels('Dataset/Faces')
images = images.astype('float32') / 255.0
images = np.expand_dims(images, axis=-1)  # Ajustar a la CNN
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))
labels_encoded = to_categorical(labels_encoded, num_classes)

# División de datos y PCA
train_images, test_images, train_labels, test_labels = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
pca = PCA(n_components=150)
train_images_flat = train_images.reshape(train_images.shape[0], -1)
train_images_pca = pca.fit_transform(train_images_flat)
test_images_flat = test_images.reshape(test_images.shape[0], -1)
test_images_pca = pca.transform(test_images_flat)
train_images_pca = np.reshape(train_images_pca, (-1, 32, 32, 1))
test_images_pca = np.reshape(test_images_pca, (-1, 32, 32, 1))

# Definir y compilar el modelo
model = models.Sequential([Input(shape=(32, 32, 1)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento y evaluación
history = model.fit(train_images_pca, train_labels, epochs=50, validation_data=(test_images_pca, test_labels))
test_loss, test_acc = model.evaluate(test_images_pca, test_labels, verbose=2)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

# Guardar el modelo
model.save('modelo.h5')
print("Modelo guardado.")

# Visualizar el entrenamiento
plt.figure(figsize=(15, 5))

# Subplot para la precisión
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Subplot para la pérdida
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
