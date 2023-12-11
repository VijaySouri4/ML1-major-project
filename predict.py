from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_num = 1

df_test = pd.read_csv(f'dataframes/test.csv')
img_path = df_test.images[img_num]
mask_path = df_test.masks[img_num]

im1 = plt.imread(img_path)
im = cv2.resize(im1, (128, 128))
im = im/255
im = im[np.newaxis, :, :, :]

mask = plt.imread(mask_path)

global_model = load_model('global_model.h5')
prediction = global_model.predict(im)


plt.figure(figsize=(16, 16))
plt.subplot(1, 3, 1)
plt.imshow(im1)
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(prediction) > .5)
plt.show()
