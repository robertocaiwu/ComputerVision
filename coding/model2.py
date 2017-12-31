
# coding: utf-8

# In[1]:


from __future__ import division
import pickle
from keras.models import Sequential
from keras.layers import *
from keras.layers.advanced_activations import PReLU
#utilities help us transform our data
from keras.utils import *
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow as tf
from keras.utils import *


# In[2]:


#Load pre-processed trained and test data

pickle_file = 'SVHN_list_single_digit.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']

  del save  # hint to help gc free up memory
# print(valid_labels.shape)
# print(valid_labels)


# In[3]:


print('Training set and labels', len(train_dataset), (train_labels).shape)
print('Validation set and labels ', len(valid_dataset), (valid_labels).shape)
print('Test set and labels', len(test_dataset), (test_labels).shape)


def CNN(input_shape, num_of_classes):

    model = Sequential()
    #formula to compute number of parameters
    # total_params =(filter_height * filter_width * input_image_channels + 1) * number_of_filters
    model.add(Convolution2D(16, 5, 5, border_mode='same',
                            input_shape= input_shape ))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(5, 5),strides=(2, 2), border_mode='same'))
#     model.add(Dropout(.5))

    model.add(Convolution2D(32, 7, 7, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), border_mode='same'))
#     model.add(Dropout(.5))

    model.add(Convolution2D(96, 5, 5, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2), border_mode='same'))

    #fully connected laye
    model.add(Flatten())
    model.add(Dense(32))
    model.add(PReLU())
#     model.add(Dropout(32))
    model.add(Dense(24))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model




# Generate images according to batch size
def gen(dataset, labels, batch_size):

#     print "calling"
    i = 0
    images = []
    digits = []
#     print batch_size
    while True:

        images.append(dataset[i])
        digits.append(labels[i])
        i+=1
        # Generate images based on batch size
        if i == batch_size:
#             print np.array(images)
            yield np.array(images), np.array(digits)
            print len(images)
            images = []
            digits = []
            i = 0
        # Generate remaining images also
        if len(images) == 0:
            continue

        yield np.array(images), np.array(digits)


# In[8]:


model_save_path = 'TrainedModel_single.hdf5'
num_classes = 10

image_size =32, 32, 1
# print image_size
batch_size = 32
num_epochs = 5
train = gen(train_dataset, train_labels, batch_size)
# print next(train)
valid = gen(valid_dataset, valid_labels, batch_size)


# In[9]:


# next(gen(train_dataset, train_labels, batch_size))


# In[12]:


model = CNN(image_size, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
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
model.fit_generator(train,
                    samples_per_epoch=np.ceil(len(train_dataset)/batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=valid,
                    validation_steps = batch_size,
                    callbacks=model_callbacks)


# # Model

# In[11]:
