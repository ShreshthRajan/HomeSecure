import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_deep_cnn(input_shape=(224, 224, 3), num_classes=80):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(image_dir, annotation_dir):
    images = []
    labels = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                annotation_path = os.path.join(annotation_dir, os.path.splitext(file)[0] + '.npz')
                if os.path.exists(annotation_path):
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    img = img / 255.0  # Normalize
                    images.append(img)
                    data = np.load(annotation_path, allow_pickle=True)
                    labels.append(data['labels'].tolist())
    return np.array(images), labels

def train_cnn(image_dir, annotation_dir, output_model_path):
    images, labels = load_data(image_dir, annotation_dir)
    
    # Flatten the list of labels and handle multi-label format
    flat_labels = [label for sublist in labels for label in sublist]
    
    # One-hot encode labels
    unique_labels = np.unique(flat_labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    labels = [[label_map[label] for label in sublist] for sublist in labels]
    labels = np.array([np.sum(tf.keras.utils.to_categorical(label, num_classes=num_classes), axis=0) for label in labels])

    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Check shapes and data consistency
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_val shape: {x_val.shape}')
    print(f'y_val shape: {y_val.shape}')

    model = create_deep_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
    
    # Add callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_model_path, save_best_only=True, monitor='val_loss')
    ]
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=callbacks)

    model.save(output_model_path)

if __name__ == "__main__":
    train_cnn('data/processed/cam1', 'data/annotations/cam1', 'models/deep_cnn_model.h5')
    train_cnn('data/processed/cam2', 'data/annotations/cam2', 'models/deep_cnn_model.h5')
