from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

# Imports
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import PReLU
#utilities help us transform our data
from keras.utils import *
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow as tf
import matplotlib.pyplot as plt


#  CNN structure
def CNN(input_, num_classes):
    #print input_shape

    model = Sequential()
    model.add(Conv2D(16, (5, 5), border_mode='same', input_shape=input_))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    # model.add(Dropout(0.5))
    model.add(Conv2D(32, (5, 5),  border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3),  border_mode='same'))
    model.add(BatchNormalization())
    # model.add(Conv2D(64, (5, 5),  border_mode='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(96, (5, 5), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(96, (5, 5), border_mode='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

# Generate images according to batch size
def gen(dataset, labels, batch_size):

    images = []
    digits = []
    i = 0
    while True:
        images.append(dataset[i])
        digits.append(labels[i])
        i+=1
        # Generate images based on batch size
        if i == batch_size:
            yield (np.array(images), np.array(digits))
            images = []
            digits = []
            i = 0
        # Generate remaining images also
        if len(images) == 0:
            continue
        yield (np.array(images), np.array(digits))

def gender_CNN(input_shape,num_classes):
    model = Sequential()

    model.add(Convolution2D(16, 7, 7, border_mode='same',
                            input_shape=input_shape))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(512, 13, 13, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2), border_mode='same'))
    model.add(Dropout(.5))

    # model.add(Conv2D(64, (5, 5),  border_mode='same'))
    # model.add(BatchNormalization())
    # # model.add(Conv2D(64, (5, 5),  border_mode='same'))
    # # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    # model.add(Dropout(0.5))

    # model.add(Conv2D(64, (3, 3),  border_mode='same'))
    # model.add(BatchNormalization())
    # # model.add(Conv2D(64, (5, 5),  border_mode='same'))
    # # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    # model.add(Dropout(0.5))



    model.add(Flatten())
    # model.add(Dense(1028))
    # model.add(PReLU())
    # model.add(Dropout(0.5))
    model.add(Dense(1028))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    pickle_file = 'dataSet_listSvhn.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        print('Training set and labels', len(train_dataset), (train_labels.shape))
        print('Validation set and labels ', len(valid_dataset), len(valid_labels))
        print('Test set and labels', len(test_dataset), len(test_labels))
        del save  # hint to help gc free up memory

    model_save_path = 'TrainedModel.hdf5'
    num_classes = 10

    image_size =32, 32, 1
    print(image_size)

    batch_size = 64
    num_epochs = 200
    #print "traind ata ", train_dataset
    train = gen(train_dataset, train_labels, batch_size)

    #train_x, train_y = (gen(train_dataset, train_labels, batch_size))
    print("training generator being called")

    valid = gen(valid_dataset, valid_labels, batch_size)
    #test_x, test_y = (gen(test_dataset, test_labels, batch_size))
    #test = (gen(test_dataset, test_labels, batch_size))


    #print("training generator being called")

    print("network being called")
    model = gender_CNN(image_size, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])

    print(model.summary())
    csv_logger = CSVLogger('training.log')
    early_stop = EarlyStopping('val_acc', patience=200, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path,
                                        'val_acc', verbose=0,
                                        save_best_only=True)

    model_callbacks = [early_stop, model_checkpoint, csv_logger]
    # print "len(train_dataset) ", len(train_dataset)
    print("int(len(train_dataset)/batch_size) ", int(len(train_dataset)/batch_size))
    K.get_session().run(tf.global_variables_initializer())

    model.fit_generator(train,
              samples_per_epoch=np.ceil(len(train_dataset)/batch_size),
# batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_data=valid,
	      validation_steps=batch_size,
              callbacks=model_callbacks)

    score = model.evaluate(test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    history = AccuracyHistory()
    plt.plot(range(1,11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.save('acc.png')

    print('done training')
