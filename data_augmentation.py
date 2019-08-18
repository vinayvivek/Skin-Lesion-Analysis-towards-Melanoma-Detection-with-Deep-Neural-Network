# Import the libraries
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil

# Create a new directory for the images
base_dir = 'core_data'
os.mkdir(base_dir)

# Training file directory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train)

# Validation file directory
val_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation)

# Create new folders in the training directory for each of the classes
nv = os.path.join(train, 'nv')
os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
os.mkdir(bcc)
ak= os.path.join(train_dir, 'ak')
os.mkdir(ak)
vasc = os.path.join(train_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
os.mkdir(df)
scc= os.path.join(train_dir, 'scc')
os.mkdir(scc)


# Create new folders in the validation directory for each of the classes
nv = os.path.join(validation, 'nv')
os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
os.mkdir(bcc)
ak= os.path.join(val_dir, 'ak')
os.mkdir(ak)
vasc = os.path.join(val_dir, 'vasc')
os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
os.mkdir(df)
scc= os.path.join(val_dir, 'scc')
os.mkdir(scc)

# Read the metadata
df = pd.read_csv('ISIC_2019_Training_Metadata_3.csv')

# Display some information in the dataset
df.head()

# Set y as the labels
y = df['dx']

# Split the metadata into training and validation
df_train, df_val = train_test_split(df, test_size=0.1, random_state=101, stratify=y)

# Print the shape of the training and validation split
print(df_train.shape)
print(df_val.shape)

# Find the number of values in the training and validation set
df_train['dx'].value_counts()
df_val['dx'].value_counts()

# Transfer the images into folders
# Set the image id as the index
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder= os.listdir('ISIC_2019_Training_Input')

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

# Transfer the training images
for image in train_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    # source path to image
    src = os.path.join('ISIC_2019_Training_Input', fname)
    # destination path to image
    dst = os.path.join(train, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

# Transfer the validation images
for image in val_list:

    fname = image + '.jpg'
    label = df.loc[image, 'dx']

    # source path to image
    src = os.path.join('ISIC_2019_Training_Input', fname)
    # destination path to image
    dst = os.path.join(validation, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)

# Check how many training images are in each folder
print(len(os.listdir('core_data/train/nv')))
print(len(os.listdir('core_data/train/mel')))
print(len(os.listdir('core_data/train/bkl')))
print(len(os.listdir('core_data/train/bcc')))
print(len(os.listdir('core_data/train/ak')))
print(len(os.listdir('core_data/train/vasc')))
print(len(os.listdir('core_data/train/df')))
print(len(os.listdir('core_data/train/scc')))

# Check how many validation images are in each folder
print(len(os.listdir('core_data/validation/nv')))
print(len(os.listdir('core_data/validation/mel')))
print(len(os.listdir('core_data/validation/bkl')))
print(len(os.listdir('core_data/validation/bcc')))
print(len(os.listdir('core_data/validation/ak')))
print(len(os.listdir('core_data/validation/vasc')))
print(len(os.listdir('core_data/validation/df')))
print(len(os.listdir('core_data/validation/scc')))

# Augment the data
# Class 'nv' is not going to be augmented
class_list = ['mel', 'bkl', 'bcc', 'ak', 'vasc', 'df','scc']

for item in class_list:

    # Create a temporary directory for the augmented images
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)

    # Create a directory within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # List all the images in the directory
    img_list = os.listdir('core_dir/train/' + img_class)

    # Copy images from the class train dir to the img_dir
    for fname in img_list:
        # source path to image
        src = os.path.join('core_data/train/' + img_class, fname)
        # destination path to image
        dst = os.path.join(img_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = aug_dir
    save_path = 'core_data/train/' + img_class

    # Create a data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 100

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 10000  # total number of images we want to have in each class
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 10000 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')

# Check how many train images are each folder (original + augmented)
print(len(os.listdir('core_data/train/nv')))
print(len(os.listdir('core_data/train/mel')))
print(len(os.listdir('core_data/train/bkl')))
print(len(os.listdir('core_data/train/bcc')))
print(len(os.listdir('core_data/train/ak')))
print(len(os.listdir('core_data/train/vasc')))
print(len(os.listdir('core_data/train/df')))
print(len(os.listdir('core_data/train/scc')))

# Check how many validation images are in each folder
print(len(os.listdir('core_data/validation/nv')))
print(len(os.listdir('core_data/validation/mel')))
print(len(os.listdir('core_data/validation/bkl')))
print(len(os.listdir('core_data/validation/bcc')))
print(len(os.listdir('core_data/validation/ak')))
print(len(os.listdir('core_data/validation/vasc')))
print(len(os.listdir('core_data/validation/df')))
print(len(os.listdir('core_data/validation/scc')))
