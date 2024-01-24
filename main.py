# Import necessary libraries
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_your_dataset():
    # Define the path to your dataset
    dataset_path = 'path_to_your_dataset'

    # Initialize an image data generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load training data
    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    # Load validation data
    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model():
    # Load your dataset
    X, y = load_your_dataset()

    # Split your data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Define your model architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile your model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train your model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Save your model
    model.save('deepfake_detection_model.h5')

def detect_deepfake(video_path):
    # Load pre-trained model for deepfake detection
    model = models.load_model('deepfake_detection_model.h5')

    # Load video
    video = cv2.VideoCapture(video_path)

    while True:
        # Read video frame
        ret, frame = video.read()

        if not ret:
            break

        # Preprocess frame for model input
        frame = cv2.resize(frame, (128, 128))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Predict if the frame is real or fake
        prediction = model.predict(frame)

        if prediction < 0.5:
            print('Real')
        else:
            print('Fake')

    video.release()
    cv2.destroyAllWindows()

# Train the model
train_model()

# Test the function
detect_deepfake('path_to_your_video.mp4')
