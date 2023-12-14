from PIL import Image
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_city_name(string):
    pattern = re.compile('[a-zA-Z]+')
    matches = pattern.findall(string)
    result = ''.join(matches)
    return result

def crop(image_path, image_name):
    
    tif_image = Image.open(image_path)
    cropped_image_size = 250
    width, height = tif_image.size

    num_cropped_images_width = width // cropped_image_size
    num_cropped_images_height = height // cropped_image_size

    for i in range(num_small_images_height):
        for j in range(num_small_images_width):
            left = j * cropped_image_size
            upper = i * cropped_image_size
            right = left + cropped_image_size
            lower = upper + cropped_image_size

            cropped_image = tif_image.crop((left, upper, right, lower))
            city = extract_city_name(image_name)
            
            save_path = f'test_cropped/images/{city}/{image_name}_{i}_{j}.jpg'
            cropped_image.save(save_path)

def crop_gt(image_path, image_name):
    
    tif_image = Image.open(image_path)
    
    cropped_image_size = 250
    width, height = tif_image.size

    num_cropped_images_width = width // cropped_image_size
    num_cropped_images_height = height // cropped_image_size

    for i in range(num_small_images_height):
        for j in range(num_small_images_width):
            left = j * cropped_image_size
            upper = i * cropped_image_size
            right = left + cropped_image_size
            lower = upper + cropped_image_size

            cropped_image = tif_image.crop((left, upper, right, lower))
            city = extract_city_name(image_name)
            
            
            save_path = f'train_cropped/gt/{city}/{image_name}_{i}_{j}.jpg'
            cropped_image.save(save_path)

path = "train_og/images"
for image in tqdm(os.listdir(path)):
    img = image.split(".")
    crop(f'{path}/{image}', img[0])

path = "train_og/gt"
for image in tqdm(os.listdir(path)):
    img = image.split(".")
    crop_gt(f'{path}/{image}', img[0])

path = "test_og/images"
for image in tqdm(os.listdir(path)):
    img = image.split(".")
    crop(f'{path}/{image}', img[0])

def is_valid(arr):
    flat_matrix = arr.flatten()
    unique_values, counts = np.unique(flat_matrix, return_counts=True)
    frequency_dict = dict(zip(unique_values, counts))
    if 255 in frequency_dict and frequency_dict[255] > 625:
        return True
    return False

for city in ['austin', 'chicago', 'kitsap', 'vienna', 'tyrolw']:
    
    images = []
    masks = []
    
    images_temp = os.listdir(f"train_cropped/images/{city}")
    masks_temp = os.listdir(f"train_cropped/gt/{city}")
    
    for image_path, mask_path in zip(images_temp, masks_temp):
        mask = plt.imread(f"train_cropped/gt/{city}/{mask_path}")
        image_array = np.array(mask)
        threshold_value = 125
        image_array[image_array > threshold_value] = 255
        image_array[image_array <= threshold_value] = 0
        if is_valid(image_array):
            images.append(f"train_cropped/images/{city}/{image_path}") 
            masks.append(f"train_cropped/gt/{city}/{mask_path}")

    combined_lists = list(zip(images, masks))
    random.shuffle(combined_lists)
    
    images, masks = zip(*combined_lists)
    
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state = 42)
    
    print(f"{city}")
    df = pd.DataFrame()
    df['images'] = train_images
    df['masks'] = train_masks
    print(f"Train len: {len(df)}")
    df.to_csv(f"dataframes/{city}_train.csv")
    
    df = pd.DataFrame()
    df['images'] = val_images
    df['masks'] = val_masks
    print(f"Test len: {len(df)}")
    df.to_csv(f"dataframes/{city}_val.csv")
