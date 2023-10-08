import numpy as np
import tensorflow as tf
import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model


NUM_CLASSES = 10
IMG_SIZE = 224
new_custom_dataset_path = 'C:/Users/ASUS/Desktop/HPS/Splited_Datasets/test_data'


def build_fine_tuned_model(num_classes, base_model_weights_path):
    # Load the entire model including architecture and weights
    base_model = load_model(base_model_weights_path)

    # Freeze the base model's layers
    base_model.trainable = False

    # Rebuild top
    x = layers.Flatten(name="flatten")(base_model.layers[-2].output)  # Use second-to-last layer's output
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Build the new model with base_model.input as input
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="FineTunedEfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # You can adjust the learning rate
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def build_base_model(base_model_weights_path):
    # Load the entire model including architecture and weights
    base_model = load_model(base_model_weights_path)

    # Set all layers in the base_model as trainable
    for layer in base_model.layers:
        layer.trainable = True

    # Compile the model
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # You can adjust the learning rate
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return base_model


model = build_base_model(   './model_result/V8_60ep_custom_aug.h5')

dataset_path_custom = os.listdir(new_custom_dataset_path)
print (dataset_path_custom)  #what kinds of classes are in this dataset

print("Types of classes labels found: ", len(dataset_path_custom))

class_labels = []
for item in dataset_path_custom:
 # Get all the file names
 all_classes = os.listdir(new_custom_dataset_path + '/' +item)
 #print(all_classes)

 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))
    #print(class_labels[:5])

# Build a dataframe        
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
print("Total number of images in the dataset: ", len(df))
label_count = df['Labels'].value_counts()
print(label_count)

dataset_path = os.listdir(new_custom_dataset_path)

im_size = 224

images = []
labels = []

for i in dataset_path_custom:
    data_path = new_custom_dataset_path + '/' + str(i)  
    filenames = [i for i in os.listdir(data_path) ]
   
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

images = np.array(images)
images = images.astype('float32') / 255.0

y=df['Labels'].values
print(y)
sorted_indices = np.argsort([s[0] for s in y])
y = y[sorted_indices]
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)

y=y.reshape(-1,1)
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y).toarray()


images, Y = shuffle(images, Y, random_state=1)
train_x, train_y = images, Y  

print(train_x.shape)
print(train_y.shape)

hist = model.fit(train_x, train_y, epochs=3, verbose=2,validation_data=None)
model.save("V9_finetuned.h5")
