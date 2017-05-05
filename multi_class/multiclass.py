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

DATA_DIR = '/home/kbalak18/keras_cnn/data/PlantVillage/crowdai'

WIDTH = 224
HEIGHT = 224

x_train = []
y_train = []
x_test = []
y_test = []



nimgs = 0
for img_path in glob.glob(DATA_DIR + '/*/*'):
 nimgs += 1

print "total number of images = ", nimgs


nclass = 38


ximgs = np.zeros((nimgs,WIDTH,HEIGHT,3),dtype=np.float)
yimgs = np.zeros((nimgs,nclass),dtype=np.float)


i = 0
for img_path in glob.glob(DATA_DIR + '/*/*'):
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 img = img.astype(float)
 ximgs[i,:,:,:] = img[:,:,:]/255.0
 img_path = img_path.strip()
 j = img_path.split('/')[7].split('_')[1]
 yimgs[i,j] = 1.0
 i += 1


ind = np.arange(ximgs.shape[0])
np.random.shuffle(ind)
ximgs = ximgs[ind,:,:,:]
yimgs = yimgs[ind]


ntrain = int(nimgs*0.8)
ntest = nimgs - ntrain

x_train = ximgs[0:ntrain,:,:,:]
y_train = yimgs[0:ntrain,:]
x_test = ximgs[ntrain:nimgs,:,:,:]
y_test = yimgs[ntrain:nimgs,:]



print x_train.shape
print y_train.shape



# set base model 1 = VGG16; 2 = ResNet50
baseModel = 2

if baseModel == 1:
 print "Base Model = VGG16 "
 base_model = VGG16(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))
elif baseModel == 2:
 print "Base Model = ResNet50 "
 base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))  



x = base_model.output 
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(nclass,activation='sigmoid')(x)

    
#combine conv and fc  
model = Model(inputs=base_model.input, outputs=preds)

for layer in base_model.layers:
 layer.trainable = False


model.summary()


sgd = SGD(lr=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

nepochs = 10
nbatch = 4
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
 



