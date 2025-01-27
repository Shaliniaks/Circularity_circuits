# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:58:27 2024

@author: mu00122
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf

# Print available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define base model creation functions
def create_base_model(base_model_fn, input_size=(448, 448)):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:-4]:  # Unfreeze last few layers
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define function to save training curves
def save_training_curves(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training curves saved as {filename}")
    plt.show()

# Function to train base models
def train_base_models(train_data_dir, val_data_dir):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(448, 448), batch_size=64, class_mode='categorical', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, target_size=(448, 448), batch_size=64, class_mode='categorical', shuffle=False
    )

    # Create models
    resnet_model = create_base_model(ResNet50)
    densenet_model = create_base_model(DenseNet201)
    efficientnet_model = create_base_model(EfficientNetB0)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # Train models
    resnet_history = resnet_model.fit(train_generator, validation_data=val_generator, epochs=500, callbacks=[early_stopping, lr_scheduler])
    save_training_curves(resnet_history, "resnet_training_curves.png")
    resnet_model.save_weights('resnet_model_weights.h5')

    densenet_history = densenet_model.fit(train_generator, validation_data=val_generator, epochs=500, callbacks=[early_stopping, lr_scheduler])
    save_training_curves(densenet_history, "densenet_training_curves.png")
    densenet_model.save_weights('densenet_model_weights.h5')

    efficientnet_history = efficientnet_model.fit(train_generator, validation_data=val_generator, epochs=500, callbacks=[early_stopping, lr_scheduler])
    save_training_curves(efficientnet_history, "efficientnet_training_curves.png")
    efficientnet_model.save_weights('efficientnet_model_weights.h5')

    return resnet_model, densenet_model, efficientnet_model, train_generator.class_indices

# Confusion matrix visualization using matplotlib
def save_confusion_matrix(y_true, y_pred, labels, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    # Labeling the axes and the plot
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    # Adding text annotations inside the squares
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Title and labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved as {filename}")
    plt.show()

# Meta-model feature extraction
def extract_features(data_dir, models, preprocess_fns, class_indices):
    features = []
    labels = []

    for class_name, class_idx in class_indices.items():
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(448, 448))
            img_array = image.img_to_array(img) / 255.0

            predictions = []
            for model, preprocess_fn in zip(models, preprocess_fns):
                input_tensor = preprocess_fn(np.expand_dims(img_array, axis=0))
                predictions.append(model.predict(input_tensor).flatten())

            features.append(np.concatenate(predictions))
            labels.append(class_idx)

    return np.array(features), np.array(labels)

def main():
    train_dir = 'C:/Users/mu00122/Shalini/CC/output/Training'
    val_dir = 'C:/Users/mu00122/Shalini/CC/output/validation'

    # Train base models
    resnet_model, densenet_model, efficientnet_model, class_indices = train_base_models(train_dir, val_dir)

    # Extract meta-model training features
    preprocess_fns = [resnet_preprocess_input, densenet_preprocess_input, efficientnet_preprocess_input]
    train_features, train_labels = extract_features(train_dir, [resnet_model, densenet_model, efficientnet_model], preprocess_fns, class_indices)

    # Train meta-model
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(train_features, train_labels)
    joblib.dump(meta_model, "meta_model.pkl")

    # Extract features from the validation set
    val_features, val_labels = extract_features(val_dir, [resnet_model, densenet_model, efficientnet_model], preprocess_fns, class_indices)

    # Predict using the meta-model
    y_pred = meta_model.predict(val_features)

    # Save confusion matrix for validation data
    save_confusion_matrix(val_labels, y_pred, list(class_indices.keys()), "meta_model_confusion_matrix.png")
    print(classification_report(val_labels, y_pred, target_names=list(class_indices.keys())))

if __name__ == "__main__":
    main()
