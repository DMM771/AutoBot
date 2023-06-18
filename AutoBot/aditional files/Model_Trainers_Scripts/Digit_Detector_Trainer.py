import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

src_dir = r"Datasets/Digit_Data_Binary_new"

X = []
y = []

# Iterate over the subdirectories in the source directory
for subdir in os.listdir(src_dir):
    # Iterate over the files in the subdirectory
    for filename in os.listdir(os.path.join(src_dir, subdir)):
        # Read the image
        img = cv2.imread(os.path.join(src_dir, subdir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (39, 57))  # Resize the image to match the input shape of the model
        img = img.astype('float32') / 255  # Normalize the image
        X.append(img)
        y.append(int(subdir))  # Use the name of the subdirectory as the label

X = np.array(X)
y = np.array(y)

# Reshape X to match the input shape of the model
X = X.reshape(X.shape[0], 39, 57, 1)

# One-hot encoding
class_count = 10
encoded_y = np_utils.to_categorical(y, class_count)

# Splitting the dataset
split_ratio = 0.2
random_seed = 42
X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_y, test_size=split_ratio, random_state=random_seed)


# Model Architecture
def create_model():
    model = Sequential()

    # First Conv2D layer
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(39, 57, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    layers_config = [
        (64, (3, 3)),
        (128, (3, 3)),
        (256, (3, 3))
    ]

    for (filters, kernel_size) in layers_config:
        model.add(Convolution2D(filters, kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

    model.add(Flatten())

    dense_layers = [128, 64]
    for units in dense_layers:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

    model.add(Dense(class_count))
    model.add(Activation('softmax'))

    return model


lr = 0.001
opt = Adam(learning_rate=lr)


# Instantiate and compile the model
my_model = create_model()
my_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=7,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.1
)

datagen.fit(X_train)

# Training
epochs = 8
batch_size = 32

# Training
history = my_model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                       validation_data=(X_test, Y_test),
                       epochs=epochs, shuffle=True, verbose=1)

# Evaluation
score = my_model.evaluate(X_test, Y_test)
print('Evaluation loss: ', score[0])
print('Evaluation accuracy: ', score[1])

# Print losses and accuracies per epoch
for i in range(epochs):
    print(f"Epoch {i+1}/{epochs}")
    print(f"Train loss: {history.history['loss'][i]}, Train accuracy: {history.history['accuracy'][i]}")
    print(f"Validation loss: {history.history['val_loss'][i]}, Validation accuracy: {history.history['val_accuracy'][i]}")

# Save the model
my_model.save('Digit_my_model.h5')
