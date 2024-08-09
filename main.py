import os

import tensorflow as tf

from model.models import model
from tools import train_model, load_image
from config import *

if __name__ == '__main__':
    train_model(model)
    while True:
        n = input('Input your number of picture:')
        if n.lower() in ['e','q','exit','quit']:
            break
        img = load_image(n)
        # Parameter img is a two-dimensional array, but model only accepts three-dimensional arrays
        # Expand dimensions before give it to model
        x_predict = img[tf.newaxis, ...]

        result = model.predict(x_predict)
        pred = tf.argmax(result, axis=1)
        print(pred)

