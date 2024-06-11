import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, Input
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def load_and_preprocess_images(directory):
    images = []
    labels = []
    label_names = [name for name in os.listdir(directory) if not name.startswith('.')]
    label_map = {name: idx for idx, name in enumerate(label_names)}
    
    for label in label_names:
        person_dir = os.path.join(directory, label)
        if os.path.isdir(person_dir):
            for image_filename in os.listdir(person_dir):
                if not image_filename.startswith('.'):
                    image_path = os.path.join(person_dir, image_filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (32, 32))
                        images.append(image)
                        labels.append(label_map[label])

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels, dtype="int32")
    num_classes = len(label_names)
    return images, labels, label_map, num_classes


#imprimir el mapa de clases
def print_class_map(label_map):
    print("Mapa de Clases:")
    for label, index in sorted(label_map.items(), key=lambda item: item[1]):
        print(f"{index}: {label}")


# Cargar imágenes
dataset_path = 'Dataset/Faces'
images, labels, label_map, num_classes = load_and_preprocess_images(dataset_path)
print("Número de clases:", num_classes)
print_class_map(label_map)

# Dividir los datos en entrenamiento y prueba
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)


model = models.Sequential([
    Input(shape=(32, 32, 3)),
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=40, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

model.save('path_to_my_model.h5')  # Guarda el modelo
print("Modelo guardado en 'path_to_my_model.h5'.")

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


