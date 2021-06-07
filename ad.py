import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from CNN_extras import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from collections import Counter
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#  Global variables placed in one location for convenience while fine tuning
BASE_DIR = 'Alzheimer_s Dataset/alz_data'
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = [176, 176]
TEST_SIZE = 0.2
BATCH_SIZE = 20
EPOCHS = 100
OPTIMIZER = RMSprop(lr=0.001, rho=0.9)
LOSS = "categorical_crossentropy"
METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.CategoricalAccuracy(name='acc'), f1_score]


# Generic 2D convolution block:
def conv_block(filters, kernel_size, strides, activation='relu'):
    block = Sequential([
        Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
        Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, padding='same'),
        BatchNormalization(),
        MaxPooling2D()])
    return block


# Generic Dense block
def dense_block(units, dropout_rate, activation='relu'):
    block = Sequential([
        Dense(units=units, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate)])
    return block


def preprocessing():
    IDG = ImageDataGenerator(rescale=1./255)
    data = IDG.flow_from_directory(directory=BASE_DIR,
                                   target_size=IMG_SIZE,
                                   batch_size=7000,
                                   color_mode="grayscale",
                                   class_mode="sparse")

    data_X, data_y = data.next()

    train_X, test_X, train_y, test_y = train_test_split(data_X,
                                                        data_y,
                                                        stratify=data_y,
                                                        test_size=0.2,
                                                        random_state=124)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                                                          train_y,
                                                          stratify=train_y,
                                                          test_size=0.2,
                                                          random_state=124)

    print("Train: ", train_X.shape, train_y.shape)
    print("Valid: ", valid_X.shape, valid_y.shape)
    print("Test: ", test_X.shape, test_y.shape)

    counter = Counter(train_y)
    print("Before: ", counter)

    over_sample = ADASYN(random_state=124)
    train_X, train_y = over_sample.fit_resample(train_X.reshape(-1, 176*176), train_y)
    train_X = train_X.reshape(-1, 176, 176)
    train_X = np.repeat(train_X[..., np.newaxis], 1, -1)

    counter = Counter(train_y)
    print("After: ", counter)
    print("Train: ", train_X.shape, train_y.shape)
    print("Valid: ", valid_X.shape, valid_y.shape)
    print("Test: ", test_X.shape, test_y.shape)

    train_y = tf.one_hot(train_y, NUM_CLASSES)
    valid_y = tf.one_hot(valid_y, NUM_CLASSES)
    test_y = tf.one_hot(test_y, NUM_CLASSES)

    print(train_y.shape)

    return train_X, train_y, valid_X, valid_y, test_X, test_y


def build_model():
    model = Sequential([
        Conv2D(filters=16, kernel_size=3, strides=2, activation='relu', padding="same", input_shape=(176, 176, 1)),
        Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', padding="same"),
        Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding="same"),
        BatchNormalization(),
        MaxPooling2D(),
        conv_block(32, 3, 1),
        conv_block(64, 3, 1),
        Dropout(0.3),
        conv_block(128, 3, 1),
        Dropout(0.2),
        conv_block(256, 3, 1),
        GlobalAveragePooling2D(),
        dense_block(512, 0.6),
        dense_block(256, 0.4),
        dense_block(128, 0.3),
        dense_block(64, 0.2),
        Dense(len(CLASSES), activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    return model


def train_model(train_X, train_y, valid_X, valid_y):
    model = build_model()

    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_alzheimer_model5.h5", save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    rop = tf.keras.callbacks.ReduceLROnPlateau(patience=5)

    history = model.fit(train_X, train_y,
                        validation_data=(valid_X, valid_y),
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        callbacks=[checkpoint, rop],
                        epochs=EPOCHS)

    return history, model


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = preprocessing()
    hist, alz_model = train_model(train_data, train_labels, valid_data, valid_labels)

    fig, ax = plt.subplots(1, 3, figsize=(30, 5))
    ax = ax.ravel()

    test_scores = alz_model.evaluate(test_data, test_labels)

    for i, metric in enumerate(['loss', 'acc', 'auc']):
        ax[i].plot(hist.history[metric])
        ax[i].plot(hist.history['val_' + metric])
        ax[i].set_title(f'Model {metric}')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[i].legend(['train', 'val'])
    plt.show()

    print(f"Testing Accuracy: {test_scores[2]*100}")

    alz_model.save("alz_model5.h5")
