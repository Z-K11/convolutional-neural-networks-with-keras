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
print(ytrain.shape)
xtest = xtest.reshape(xtest.shape[0],28,28,1).astype('float32')
# normalize the pixel values between 0 and 1 
xtest=xtest/255
xtrain=xtrain/255
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)
# the to_categorical function performs one hot encoding 
num_of_classes = ytrain.shape[1]
def convolutional_model():
    model=Sequential()
    # Initializing the model
    model.add(Input(shape=(28,28,1)))
    # Add the input layer which is an image of width 28, height 28 and greyscale as channel is 1
    model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu'))
    '''Adds a convolution layer that applies 16 different convolutional filters stride here means
    that we move 1 pixel vertically or horizontally '''
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    '''Adds a pooling layer using the max pooling function the pool window size is 2x2 and stride
    means we move to pixels vertically and horizontally'''
    model.add(Flatten())
    '''flattens the output of the network to fed into the dense layers'''
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model=convolutional_model()
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=200,verbose=2)
'''Batch size means that the models instead of reading the entire data sets groups the data set
into groups of 200 chunks and after each batch the model updates its weights using backpropogation
and gradient descent helpful when dealing with large datasets '''
scores = model.evaluate(xtest,ytest,verbose=2)
print('Accuracy {} \n Error {}'.format(scores[1],100-scores[1]*100))
def _convolutional_layer():
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(8,(2,2),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_of_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model= _convolutional_layer()
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=200,verbose=2)
scores = model.evaluate(xtest,ytest,verbose=2)
print('2 Convolution and pooling layers \n Accuracy {} \n Errors{}'
.format(scores[1],100-scores[1]*100))