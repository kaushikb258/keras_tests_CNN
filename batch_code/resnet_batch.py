import cv2
import glob
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50


# Generate dummy data

DATA_DIR = '/home/kbalak18/keras_cnn/data/train/'

WIDTH = 224
HEIGHT = 224

x_train = []
y_train = []
x_test = []
y_test = []




x_train = np.zeros((16000,WIDTH,HEIGHT,3),dtype=np.float)
x_test = np.zeros((4000,WIDTH,HEIGHT,3),dtype=np.float)
y_train = np.zeros((16000),dtype=np.float)
y_test = np.zeros((4000),dtype=np.float)



for i in range(8000):
 img_path = DATA_DIR + 'cat.' + str(i) + '.jpg' 
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 img = img.astype(float)
 x_train[i,:,:,:] = img[:,:,:]/255.0
 y_train[i] = 1.0

for i in range(8000):
 img_path = DATA_DIR + 'dog.' + str(i) + '.jpg' 
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 img = img.astype(float)
 x_train[8000+i,:,:,:] = img[:,:,:]/255.0
 y_train[8000+i] = 0.0 

for i in range(2000):
 img_path = DATA_DIR + 'cat.' + str(8000+i) + '.jpg' 
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 img = img.astype(float)
 x_test[i,:,:,:] = img[:,:,:]/255.0
 y_test[i] = 1.0

for i in range(2000):
 img_path = DATA_DIR + 'dog.' + str(8000+i) + '.jpg' 
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 img = img.astype(float)
 x_test[2000+i,:,:,:] = img[:,:,:]/255.0
 y_test[2000+i] = 0.0



x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


ind = np.arange(x_train.shape[0])
np.random.shuffle(ind)
x_train = x_train[ind,:,:,:]
y_train = y_train[ind]


print x_train.shape
print y_train.shape

num_classes = 1


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))

  
x = base_model.output 
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(num_classes,activation='sigmoid')(x)

    
#combine conv and fc  
model = Model(inputs=base_model.input, outputs=preds)

for layer in base_model.layers:
 layer.trainable = False


model.summary()

sgd = SGD(lr=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

nepochs = 10
nbatch = 5
nb = x_train.shape[0]/nbatch
nv = x_test.shape[0]/nbatch

print "nb, nv = ", nb, nv


print "size of x_train = ", x_train.nbytes/(1.0e9), " GB"
print "size of x_test = ", x_test.nbytes/(1.0e9), " GB"


for e in range(nepochs):
 print "-----------------------------------------------"
 print "epoch = ", e
 for b in range(nbatch): 
  print "batch = ", b
  model.fit(x_train[b*nb:(b+1)*nb,:,:,:], y_train[b*nb:(b+1)*nb], batch_size=20, epochs=1, verbose=2, validation_data=(x_test, y_test), shuffle=True)
 

y_predict = model.predict(x_test,batch_size=1)



accuracy = 0.0
for i in range(y_test.shape[0]):
 if (y_test[i] > 0.5 and y_predict[i] > 0.5): 
  accuracy += 1.0
 elif (y_test[i] < 0.5 and y_predict[i] < 0.5):
  accuracy += 1.0

accuracy = accuracy/np.float(y_test.shape[0])
print "accuracy = ", accuracy


