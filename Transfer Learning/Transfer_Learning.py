from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg19 import VGG19
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
import cv2

#%%
(x_train,y_train),(x_test,y_test)=cifar10.load_data();
print(x_train.shape)
numberOfclass=10

y_test=to_categorical(y_test,numberOfclass)

y_train=to_categorical(y_train,numberOfclass)
input_shape=x_train.shape[1:]
#%%
plt.imshow(x_train[105].astype(np.uint8))
plt.axis("off")
plt.show()
#%%
def resize_img(img):
    numberOfImage=img.shape[0]
    new_array=np.zeros((numberOfImage,48,48,3))
    for i in range(numberOfImage):
        new_array[i]=cv2.resize(img[i,:,:,:],(48,48))
    return new_array

x_train=resize_img(x_train)
x_test=resize_img(x_test)
plt.figure()
plt.imshow(x_train[105].astype(np.uint8))
plt.axis("off")
plt.show()
#%%
vgg=VGG19(include_top=False,weights="imagenet",input_shape=(48,48,3))

vgg_layer_list=vgg.layers
print(vgg_layer_list)
model=Sequential()
for layer in vgg_layer_list:
    model.add(layer)
for i in model.layers:
    i.trainable=False
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(numberOfclass,activation="softmax"))
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=1000,epochs=5,validation_split=0.2)

















