import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras_tuner import Hyperband, Objective
import keras_tuner as kt

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
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure correct channel reading
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (32, 32))
                        images.append(image)
                        labels.append(label_map[label])

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels, dtype="int32")
    num_classes = len(label_names)
    return images, labels, label_map, num_classes

# Load images
dataset_path = 'Dataset/Faces'
images, labels, label_map, num_classes = load_and_preprocess_images(dataset_path)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

def build_model(hp):
    model = models.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Conv2D(hp.Int('conv_1_filters', min_value=32, max_value=128, step=32), (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(hp.Int('conv_2_filters', min_value=64, max_value=256, step=64), (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(hp.Int('conv_3_filters', min_value=128, max_value=512, step=128), (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.7, step=0.1)),

        layers.Conv2D(hp.Int('conv_4_filters', min_value=128, max_value=512, step=128), (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(hp.Int('conv_5_filters', min_value=256, max_value=1024, step=256), (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(hp.Int('dense_units', min_value=256, max_value=512, step=128), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Configure Hyperband tuner
tuner = kt.GridSearch(
    build_model,
    objective="val_accuracy",
    max_trials=30,
    executions_per_trial=1,
    directory='keras_tuner_hyperband',
    project_name='face_recognition'
)

# Execute the hyperparameter search
tuner.search(train_images, 
             train_labels, 
             epochs=40, 
             validation_data=(test_images, test_labels), 
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
             )

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]

# Display the best hyperparameters
print("Best hyperparameters:")
for param, value in best_hyperparameters.values.items():
    print(f"{param}: {value}")


