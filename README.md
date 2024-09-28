# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS
## Steps for Building and Training a Convolutional Neural Network (CNN)

1. **Import TensorFlow and preprocessing libraries**
2. **Download and load the dataset**
3. **Scale the dataset** between its minimum and maximum values
4. **One-hot encode** the categorical values
5. **Split the data** into training and testing sets
6. **Build the convolutional neural network model**
7. **Train the model** using the training data
8. **Plot the performance** of the model
9. **Evaluate the model** using the testing data
10. **Fit the model** and predict for a single input

## PROGRAM

### Name: SASIDEVI.V
### Register Number: 212222230136
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import utils
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_train[0]
X_test.shape
y_train.shape
y_test.shape

single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.AvgPool2D (pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='tanh'))
model.add(layers.Dense(10, activation ='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = x_train[550]
type(img)
plt.imshow(img,cmap='gray')

img = image.load_img('image.png')
tensor_img = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(tensor_img,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/3757f016-bcbf-4be9-aa3b-ae78625745f0)![image](https://github.com/user-attachments/assets/462c88d8-b8d0-45b1-87c2-5459a5fd6151)

### Classification Report
![classification_repor](https://github.com/user-attachments/assets/7e0e0e42-d0cc-4540-964d-619db2a58bc3)


### Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/d673d1e7-4f79-4d96-85de-10f7ad4d2be7)

### New Sample Data Prediction
#### Sample Input
![image](https://github.com/user-attachments/assets/72bc7111-20a6-4d01-9f28-87d100f46948)

![output](https://github.com/user-attachments/assets/ae499735-bf9f-4932-b3b3-a78c66301ba3)

## RESULT:
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
