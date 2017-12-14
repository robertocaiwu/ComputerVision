
# coding: utf-8

# In[1]:


import h5py
import cv2
import keras
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import PIL
from PIL import Image
import pandas as pd
from random import shuffle
import os
import random
import pickle


# In[2]:


class DigitStructure:
    
    def __init__(self, _filePath):
        
        self.loadFile = self.load_datafile(_filePath)
        self.digitStruct = self.readDigitStruct(self.loadFile)
        self.name = self.digitStruct['name']
        self.boxLabels = self.digitStruct["bbox"]
        
    def load_datafile(self, filepath):
        
        return h5py.File(filepath,'r')
    
    def readDigitStruct(self, datafile):
        
        return datafile["digitStruct"]
    
    def getImageName(self, index):
        
        names = []
        for i in self.loadFile[self.name[index][0]].value:
            names.append(chr(i[0]))
        return ''.join(names)
    
    def bboxExtractor(self, attr):
        
        if (len(attr) > 1):
            attr = [self.loadFile[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr
    
    # getBbox returns a dict of data for the n(th) bbox. 
    def getBboxAttributes(self,index):
        
        bbox = {}
        bb = self.boxLabels[index].item()
        bbox['height'] = self.bboxExtractor(self.loadFile[bb]["height"])
        bbox['label'] = self.bboxExtractor(self.loadFile[bb]["label"])
        bbox['left'] = self.bboxExtractor(self.loadFile[bb]["left"])
        bbox['top'] = self.bboxExtractor(self.loadFile[bb]["top"])
        bbox['width'] = self.bboxExtractor(self.loadFile[bb]["width"])
        
        return bbox
            
    def getDigitStructure(self,n):
        
        s = self.getBboxAttributes(n)
        
        s['name']=self.getImageName(n)
        return s

    # getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        
        struct = []
        for i in range(len(self.name)):
            struct.append(self.getDigitStructure(i))
        return struct
    
    def getAllDigitStructure_ByDigit(self):
        
        
        digitDictionary = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(digitDictionary)):
            item = { 'filename' : digitDictionary[i]["name"] }
            digit_labels_in_each_image = []
            for j in range(len(digitDictionary[i]['height'])):
               number = {}
               number['height'] = digitDictionary[i]['height'][j]
               number['label']  = digitDictionary[i]['label'][j]
               number['left']   = digitDictionary[i]['left'][j]
               number['top']    = digitDictionary[i]['top'][j]
               number['width']  = digitDictionary[i]['width'][j]
               digit_labels_in_each_image.append(number)
            structCnt = structCnt + 1
            item['boxes'] = digit_labels_in_each_image
            result.append(item)
        return result
    


# # Load Train Data

# In[3]:


training_dataPath = './train/'
train_digitStruct = training_dataPath + 'digitStruct.mat'
digitStructure = DigitStructure(train_digitStruct)
training_data = digitStructure.getAllDigitStructure_ByDigit()


# # Load Test Data

# In[4]:


test_dataPath = './test/'
test_digitStructure = test_dataPath + 'digitStruct.mat'
digitStructure = DigitStructure(test_digitStructure)
testing_data = digitStructure.getAllDigitStructure_ByDigit()


# In[13]:



class PreProcessDataSet:
    
    def __init__(self, data, path):
        
        self.data = data
        self.path = path
        self.image_size = (32,32,1)
        self.dataSet =                 np.ndarray([len(self.data), 32, 32, 1], dtype = 'float32')
        self.num_classes = 10
        self.total_digits = 6  #data set contains maximum 6 digits
        # initialize all elements with 10
        self.labels = np.ones([len(self.data), self.num_classes], dtype=int) * 10
        self.images = []
        self.validation_index = []
        self.training_index = []
    
    def preProcess(self):
        
        for i in np.arange(len(self.dataSet)):
            get_filename = self.data[i]["filename"]
            filename = self.path + get_filename
            read_image = Image.open(filename)
            image_size = read_image.size
            boxList = self.data[i]['boxes']
            number_of_digits = len(boxList)
            self.labels[i, 0] = number_of_digits
            
            #initalize arrays(top, left, height, width) based on num of digits
            height = np.ndarray([number_of_digits], dtype='float32') 
            width = np.ndarray([number_of_digits], dtype='float32')
            top = np.ndarray([number_of_digits], dtype='float32')
            left = np.ndarray([number_of_digits], dtype='float32')
            
            for digits in np.arange(number_of_digits):
                #if digits are less than 5
                if digits < 5:
                    self.labels[i, digits+1] = boxList[digits]['label']
#                     self.number_of_labels[i, digits+1] = boxList[digits]['label']
                    #if digit is 10, we consider it as 0
                    if boxList[digits]['label'] == 10:
                        self.labels[i, digits+1] = 0
#                         self.number_of_labels[i, digits+1] = 0
                #take index of image that has more than 5 digits
                else: print('#',i,'image has more than 5 digits.')
                    
                height[digits] = boxList[digits]['height'] 
                width[digits] = boxList[digits]['width']
                top[digits] = boxList[digits]['top']
                left[digits] = boxList[digits]['left']
                    

            #compute top left heigh and width of image
            image_top = np.amin(top)
            image_left = np.amin(left)
            image_height = np.amax(top) + height[np.argmax(top)] - image_top
            image_width = np.amax(left) + width[np.argmax(left)] - image_left
            
        
            #adjust to make them feasible for cropping
            image_top = np.floor(image_top - 0.1 * image_height)
            image_left = np.floor(image_left - 0.1 * image_width)
            image_bottom = np.amin([np.ceil(image_top + 1.2 * image_height), image_size[1]])
            image_right = np.amin([np.ceil(image_left + 1.2 * image_width), image_size[0]])
            
            
            read_image =                 read_image.crop((image_left, image_top, image_right, image_bottom)).resize([32,32], Image.ANTIALIAS)

            gray_image = read_image.convert("L")       
            gray_image = np.expand_dims(gray_image, -1)
#             print gray_image.shape
            gray_image = self.normalization(gray_image)
            
            #append gray images in list.
            self.images.append(gray_image[:,:,:])
            self.dataSet[i,:,:,:] = gray_image[:,:,:]
            
        return self.dataSet, self.labels, self.images
            
                
    def normalization(self, image):

        mean = np.mean(image, dtype='float32')
        standard_deviation = np.std(image, dtype='float32', ddof=1)

        if standard_deviation < 1e-4:
            standard_deviation = 1.

        image = (image - mean)/standard_deviation

        return image
    
    
    def createValidationSet(self, dataset, labels):
                
        
        split_portion = int(len(dataset) * 0.2)
    
        train_dataset, train_labels = self.shuffleSet(dataset, labels)
        valid_dataset = train_dataset[:split_portion]
        valid_labels = train_labels[:split_portion]
        
        train_dataset = train_dataset[split_portion:]
        train_labels = train_labels[split_portion:]
        
        return train_dataset, train_labels, valid_dataset, valid_labels
        
        
    def shuffleSet(self, data, labels ):
        
#         permutation = np.random.permutation(labels.shape[0])
#         shuffled_dataset = data[permutation,:,:]
#         shuffled_labels = labels[permutation]

        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = [data[i] for i in permutation]
        shuffled_labels = labels[permutation]
        
        return shuffled_dataset, shuffled_labels
        


# In[14]:


poces = PreProcessDataSet(training_data, training_dataPath)


# In[15]:


train_dataset, train_labels, images_list = poces.preProcess()
#train_num_labels contains 10 classes, 0 to 9 and 10(represents no digit) 


# In[16]:


print (train_dataset).shape, len(train_labels), len(images_list)


# In[20]:


len(images_list)


# In[23]:


del images_list[29929]
train_dataset = np.delete(train_dataset, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)
# train_num_labels = np.delete(train_num_labels, 29929, axis=0)


# In[25]:


poces = PreProcessDataSet(training_data, training_dataPath)


# In[26]:


train_dataset, train_num_labels, valid_dataset, valid_labels = poces.createValidationSet(images_list, train_labels)


# In[30]:


print len(train_dataset), len(train_num_labels), len(valid_dataset), len(valid_labels)


# # Generate Test Data

# In[31]:


poces = PreProcessDataSet(testing_data, test_dataPath)


# In[32]:


test_dataset, test_labels, images_test_list = poces.preProcess()
#test_dataset contains images in array
#images_test_list contains images in list
#test_labels contains labels of images in list


# In[ ]:


# save pickle file


# In[ ]:


pickle_file = 'dataSet_listSvhn.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_num_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': images_test_list,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

