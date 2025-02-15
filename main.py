from keras.models import Sequential
from keras.layers import Dense
# dense layer 
from keras.layers import Input
# input layer
from keras.utils import to_categorical
# for one hot encoding 
from keras.layers import Conv2D
# to add convolutional layer
from keras.layers import MaxPooling2D
# for pooling layer
from keras.layers import Flatten
# to flatten the data for fully connected layers
from keras.datasets import mnist
# import data
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
# reshaping data to be samples,pixels,width and height
print(xtrain.shape)
xtrain = xtrain.reshape(xtrain.shape[0],28,28,1).astype('float32')
'''reshape is in the form of [samples][width][height][channel] channel meaning if image is greyscale
or rgb'''