import keras
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import *
#utilities help us transform our data later
from keras.utils import * 
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow as tf

#%matplotlib inline

class TrainMnistDataSet:

	def __init__(self):

		self.image_height = 28
		self.image_width = 28
		self.image_depth = 1
		
	def preProcess(self, x_train, y_train, x_test, y_test):

		x_train = x_train.reshape(x_train.shape[0], self.image_height, self.image_width, self.image_depth)
		x_test = x_test.reshape(x_test.shape[0], self.image_height, self.image_width, self.image_depth)
		
		# Convert data type and normalise values

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		

		x_train /= 255
		x_test /= 255 

		# Convert 1-dimensional class arrays to 10-dimensional class matrices
		
		y_train = np_utils.to_categorical(y_train,10)
		y_test = np_utils.to_categorical(y_test,10)

		return x_train, y_train, x_test, y_test

	def modelCNN(self, input_shape, number_of_classes):


	    model = Sequential()

	    model.add(Convolution2D(16, 7, 4, border_mode='same',
	                        input_shape=input_shape))
	    model.add(PReLU())
	    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
	    model.add(BatchNormalization())

	    model.add(Convolution2D(32, 5, 3, border_mode='same'))
	    model.add(PReLU())
	    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
	    model.add(BatchNormalization())

	    model.add(Convolution2D(64, 3, 3, border_mode='same'))
	    model.add(PReLU())
	    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

	    model.add(Flatten())
	    model.add(Dense(512))
	    model.add(PReLU())
	    model.add(Dropout(0.5))
	    model.add(Dense(512))
	    model.add(PReLU())
	    model.add(Dropout(0.5))
	    model.add(Dense(num_classes))
	    model.add(Activation('softmax'))

	    return model



if __name__ == '__main__':

	trainMnist = TrainMnistDataSet()
	(x_train, y_train), (x_test, y_test) =  mnist.load_data()
	X_train, Y_train, X_test, Y_test = trainMnist.preProcess(x_train, y_train, x_test, y_test)
	input_shape = 28, 28, 1
	num_classes = 10
	nb_epoch = 6
	batch_size = 64

	model = trainMnist.modelCNN(input_shape, num_classes)
	model_save_path = 'TrainedModelMnist.hdf5'
	
	model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

	print(model.summary())
    
	csv_logger = CSVLogger('training.log')
    
	early_stop = EarlyStopping('val_acc', patience=200, verbose=1)
	model_checkpoint = ModelCheckpoint(model_save_path,
                                        'val_acc', verbose=0,
                                        save_best_only=True)
	model_callbacks = [early_stop, model_checkpoint, csv_logger]
	K.get_session().run(tf.global_variables_initializer())

	model.fit(X_train,Y_train,batch_size,nb_epoch, verbose=1,validation_data=(X_test,Y_test),
							callbacks = model_callbacks)
	