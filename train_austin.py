import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.models import Mode
import threading

from Models import Unet
from Models import AttnUnet
from Models import ResUnet
from fed_client import Client

city = 'austin'
global_size = (128, 128, 3)


df_train = pd.read_csv(f'dataframes/{city}.csv')
df_val = pd.read_csv(f'dataframes/{city}.csv')


index = 0
step = 1000
divisions = len(df_train)
batch = 16


def data_generator(df, augmentation):

    image_generator = augmentation.flow_from_dataframe(
        dataframe=df,
        x_col="images",
        batch_size=batch,
        color_mode="rgb",
        target_size=global_size[:2],
        seed=1,
        class_mode=None
    )

    mask_generator = augmentation.flow_from_dataframe(
        dataframe=df,
        x_col="masks",
        batch_size=batch,
        target_size=global_size[:2],
        color_mode="grayscale",
        seed=1,
        class_mode=None
    )

    gen = zip(image_generator, mask_generator)

    for image, mask in gen:
        image = image/255
        mask[mask <= 125] = 0
        mask[mask > 1] = 1
        yield image, mask


binary_crossentropy = tf.keras.losses.BinaryCrossentropy()


def DiceLoss(y_true, y_pred, smooth=1e-6):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - score


def requ(local_weights):
    client = Client()
    client_thread = threading.Thread(
        target=client.send_array, args=(local_weights))  # args=(np.random.rand(10),)
    client_thread.start()
    client_thread.join()  # Ensuring array is sent before receiving

    receive_thread = threading.Thread(target=client.receive_merged_array)
    receive_thread.start()
    receive_thread.join()


aug = ImageDataGenerator(horizontal_flip=True)
aug_val = ImageDataGenerator()
val_gen = data_generator(df_val, aug_val)

for i in range(0, divisions, step):
    pick = df_train.iloc[i: i+step]
    train_gen = data_generator(pick, aug)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    unet_local = Unet((global_size))
    local_model = unet_local.build()
    local_model.compile(optimizer='adam', loss=binary_crossentropy)  # metrics

    if i != 0:
        global_weights = requ(local_weigths)
        local_model.set_weights(global_weights)

    local_model.fit(train_gen, epochs=30, steps_per_epoch=len(
        pick)/16, validation_data=val_gen, validation_steps=len(df_val)/16, callbacks=[callback])
    local_weigths = local_model.get_weights()

    K.clear_session()
