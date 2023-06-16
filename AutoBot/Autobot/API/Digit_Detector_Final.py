import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('Trained_Models/Digit_my_model.h5')

# Now you can use the loaded model to make predictions
final = []


def predict_digit(img_path):
    # Load the image file
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 39x57 pixels
    img = cv2.resize(img, (39, 57))

    # Flatten the image into a 1x2213 array
    img = img.reshape(1, 39, 57, 1)

    # Normalize the image to 0-1
    img = img.astype('float32')
    img /= 255

    # Use the trained model to predict the digit
    prediction = model.predict(img, verbose=0)

    # Get the digit
    digit = np.argmax(prediction)
    final.append(digit)
    # Get the accuracy
    accuracy = np.max(prediction)
    return digit, accuracy


def get_final():
    return final


def detect(path):
    # Call the function
    digit, accuracy = predict_digit(path)

    # # Print the prediction and accuracy
    # print('Predicted digit:', digit)
    # print('Prediction accuracy:', accuracy)
    return digit, accuracy
