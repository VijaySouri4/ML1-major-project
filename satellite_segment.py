from Models import Unet
from Models import AttnUnet
from Models import ResUnet

city = 'austin'

df_train = pd.read_csv(f'dataframes/{city}.csv')
df_val = pd.read_csv(f'dataframes/{city}.csv')


index = 0
step = 1000
divisions = len(df)


def data_generator(df,augmentation):
    
    image_generator = augmentation.flow_from_dataframe(
        dataframe = df,
        x_col = "images",
        batch_size = batch,
        color_mode="rgb",
        target_size = (128,128),
        seed=1,
        class_mode= None
    )
    
    mask_generator = augmentation.flow_from_dataframe(
        dataframe = df,
        x_col = "masks",
        batch_size = batch,
        target_size = (128,128),
        color_mode="grayscale",
        seed=1,
        class_mode= None
    )
    
    gen = zip(image_generator, mask_generator)
    
    for image, mask in gen:
        image = image/255
        mask[mask <= 125] = 0
        mask[mask > 1] = 1
        yield image, mask 

aug = ImageDataGenerator(horizontal_flip = True)
aug_val = ImageDataGenerator()
val_gen = data_generator(df_val_50,aug_val)



for i in range(0, divisions, step):
    pick = df_train.iloc[i: i+step]
    train_gen = data_generator(pick,aug)
    

