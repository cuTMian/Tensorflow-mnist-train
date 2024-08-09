import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from config import *

def train_model(model):
    # Load mnist datasets
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=(x_test, y_test),
              validation_freq=1,
              callbacks=[cp_callback])
    model.summary()
    show_lines(history)

def show_lines(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def load_image(number):
    img = Image.open(PIC_FILE_PATH.format(number))
    img = img.resize((28, 28), Image.LANCZOS)
    img = np.array(img.convert('L'))

    # Input images are white-background-black-charactor number
    # But mnist datasets are black-background-white-charactor number
    # So it's necessary to reverse the color of the image

    #img = 255 - img
    for i in range(28):
        for j in range(28):
            if img[i][j] > 200:
                img[i][j] = 0
            else:
                img[i][j] = 255

    # Normalization
    img = img / 255.0
    return img


