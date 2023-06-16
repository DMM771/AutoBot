import tempfile

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model


def load_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).resize(target_size)
    img_tensor = np.array(img)  # convert image to numpy array
    img_tensor = np.expand_dims(img_tensor, axis=0)  # add a dimension for the batch
    img_tensor = img_tensor / 255.  # normalize to [0,1]
    return img_tensor


def get_color(img):
    # Load your trained model
    model_path = r"Trained_Models/color_model.h5"
    model = load_model(model_path)

    temp_path = tempfile.NamedTemporaryFile(suffix=".jpg").name
    cv2.imwrite(temp_path, img)
    img_tensor = load_image(temp_path)

    # Map of classes
    labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'silver', 'white', 'yellow']
    labels.sort()

    # Use the model to predict the image's color class
    preds = model.predict(img_tensor, verbose=0)

    class_index = np.argmax(preds[0])
    # print('Predicted class:',
    #       labels[class_index])  # Here, we use the labels list to map the index to the corresponding color
    return labels[class_index]
