import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras

from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

no_of_classes=5

def getdata(dir):
    data_dir =dir
    data = load_files(data_dir)
    X = np.array(data['filenames'])
    y = np.array(data['target'])
    labels = np.array(data['target_names'])
    # numbers are corresponding to class label. We need to change them to a vector of 5 elements.
    # Remove .pyc or .py files
    pyc_file_pos = (np.where(file == X) for file in X if file.endswith(('.pyc', '.py')))
    for pos in pyc_file_pos:
        X = np.delete(X, pos)
        y = np.delete(y, pos)
    X = np.array(convert_img_to_arr(X))
    X = X.astype('float32') / 255
    no_of_classes = len(np.unique(y))
    y = np.array(np_utils.to_categorical(y, no_of_classes))
    return X,y

#We have only the file names in X. Time to load the images from filename and save it to X.
def convert_img_to_arr(file_path_list):
    arr = []
    for file_path in file_path_list:
        img = load_img(file_path, target_size = (224,224))
        img = img_to_array(img)
        arr.append(img)
    return arr


X_train, y_train=getdata('../input/train')
X_test,y_test=getdata('../input/test')
X_train,X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.1)
# **Fine-tuning**

# Fine-tuning
from keras.models import Model
from keras import optimizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#load the VGG16 model without the final layers(include_top=False)
base_model = applications.VGG16(weights='imagenet', include_top=False)
print('Loaded model!')

#Let's freeze the first 15 layers - if you see the VGG model layers below, 
# we are freezing till the last Conv layer.
for layer in base_model.layers[:15]:
    layer.trainable = False
    
base_model.summary()


# Now, let's create a top_model to put on top of the base model(we are not freezing any layers of this model) 
top_model = Sequential()  
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(no_of_classes, activation='softmax'))
top_model.summary()


# In the summary above of our base model, trainable params is 2,565

# Let's build the final model where we add the top_model on top of base_model.
model = Sequential()
model.add(base_model)
model.add(top_model)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
# When we check the summary below,  and trainable params for model is 7,081,989 = 7,079,424 + 2,565





# Time to train our model !
epochs = 100
batch_size=32
best_model_finetuned_path = 'best_finetuned_model.hdf5'

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,y_train,
    batch_size=batch_size)

validation_generator = test_datagen.flow(
    X_valid,y_valid,
    batch_size=batch_size)
#checkpointer = ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(X_train) // batch_size,
#     epochs= epochs ,
#     validation_data=validation_generator,
#     validation_steps=len(X_valid) // batch_size,
#     callbacks=[checkpointer])


model.load_weights(best_model_finetuned_path)  
   
(eval_loss, eval_accuracy) = model.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("Loss: {}".format(eval_loss)) 

import matplotlib.pyplot as plt 
# Let's visualize the loss and accuracy wrt epochs
def plot(history):
    plt.figure(1)  

     # summarize history for accuracy  

    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  

     # summarize history for loss  

    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()
#plot(history)


# **Bottleneck features**




from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import math

epochs = 100
batch_size = 32

model = applications.VGG16(weights='imagenet', include_top=False)

datagen = ImageDataGenerator()  
   
generator = datagen.flow(  
     X_train,   
     batch_size=batch_size,    
     shuffle=False)  
   
train_data = model.predict_generator(  
     generator, int(math.ceil(len(X_train) / batch_size)) )


generator = datagen.flow(  
     X_valid,   
     batch_size=batch_size,    
     shuffle=False)  
   
validation_data = model.predict_generator(  
     generator, int(math.ceil(len(X_valid) / batch_size)) )

generator = datagen.flow(  
     X_test,   
     batch_size=batch_size,    
     shuffle=False)  
   
test_data = model.predict_generator(generator, int(math.ceil(len(X_test) / batch_size)))

from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

best_model_bottleneck_path = 'best_bottleneck_model.hdf5'

model = Sequential()  
model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
model.add(Dense(no_of_classes, activation='softmax'))  
   
model.compile(optimizer='rmsprop',  
              loss='categorical_crossentropy', metrics=['accuracy'])  
  
checkpointer = ModelCheckpoint(best_model_bottleneck_path,save_best_only = True,verbose = 1)

# history = model.fit(train_data, y_train,
#           epochs=epochs,
#           batch_size=batch_size,
#           validation_data=(validation_data, y_valid),
#           callbacks =[checkpointer])


model.load_weights(best_model_bottleneck_path)
(test_loss, test_accuracy) = model.evaluate(  
     test_data, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(test_accuracy * 100))  
print("Loss: {}".format(test_loss)) 

#plot(history)


# **Fine-tuning with bottleneck features**

from keras.models import Model
from keras import optimizers

base_model = applications.VGG16(weights='imagenet', include_top=False)

best_model_finetuned_bottleneck = 'best_bottleneck_finetuned_model.hdf5'

for layer in base_model.layers[:15]:
    layer.trainable = False

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(no_of_classes, activation='softmax'))

# loading the weights of bottle neck features model
top_model.load_weights(best_model_bottleneck_path)

model = Sequential()
model.add(base_model)
model.add(top_model)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,y_train,
    batch_size=batch_size)

validation_generator = test_datagen.flow(
    X_valid,y_valid,
    batch_size=batch_size)
#checkpointer = ModelCheckpoint(best_model_finetuned_bottleneck,save_best_only = True,verbose = 1)

# fine-tune the model
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(X_train) // batch_size,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=len(X_valid) // batch_size,
#     callbacks=[checkpointer])
model.load_weights(best_model_finetuned_bottleneck)

(test_loss, test_accuracy) = model.evaluate(
    X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Loss: {}".format(test_loss))
#plot(history)



