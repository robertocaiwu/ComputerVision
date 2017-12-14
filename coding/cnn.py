from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import pickle
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
def cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
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
            yield np.array(images), np.array(digits)
            images = []
            digits = []
        # Generate remaining images
        if i == len(dataset):
            yield np.array(images), np.array(digits)
            images, digits = [], []
            i = 0

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
    print image_size

    batch_size = 150
    num_epochs = 50

    train_x, train_y = (gen(train_dataset, train_labels, batch_size))
    print "training generator being called"

    # valid_x, valid_y = (gen(valid_dataset, valid_labels, batch_size))
    test_x, test_y = (gen(test_dataset, test_labels, batch_size))
    print "training generator being called"

    print "network being called"
    model = CNN(image_size, num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

    print model.summary()
    csv_logger = CSVLogger('training.log')
    early_stop = EarlyStopping('val_acc', patience=200, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_path,
                                        'val_acc', verbose=0,
                                        save_best_only=True)

    model_callbacks = [early_stop, model_checkpoint, csv_logger]
    # print "len(train_dataset) ", len(train_dataset)
    print "int(len(train_dataset)/batch_size) ", int(len(train_dataset)/batch_size)
    K.get_session().run(tf.global_variables_initializer())

    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=model_callbacks)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    history = AccuracyHistory()
    plt.plot(range(1,11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.save('acc.png')

    print('done training')
