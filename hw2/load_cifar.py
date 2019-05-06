#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pickle 

# In[127]:


# Step 0: load cifar data give file name e.g. "./data_batch_1"
def unpickle(file):
    """load the cifar-10 data"""
    
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


# In[128]:


#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
    """
    Args:
    folder_path: the directory contains data files
    batch_id: training batch id (1,2,3,4,5)
    Return:
    features: numpy array that has shape (10000,3072)
    labels: a list that has length 10000
    """
    ##load batch using pickle###
    file_name = folder_path + "data_batch_{}".format(batch_id)
    data_batch = unpickle( file_name )

    ###fetch features using the key ['data']###
    features = data_batch["data"]
    ###fetch labels using the key ['labels']###
    labels = data_batch["labels"]
    
    #print( "loaded ...feature shape:" , features.shape , "feature label length:", len( labels ) )
    return features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    ##load batch using pickle###
    file_name = folder_path + "test_batch"
    data_batch = unpickle( file_name )

    ###fetch features using the key ['data']###
    features = data_batch["data"]
    ###fetch labels using the key ['labels']###
    labels = data_batch["labels"]
    
    #print( "loaded ...feature shape:" , features.shape , "feature label length:", len( labels ) )
    return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""  
    this should be loaded from batches.meta
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names(folder_path):
    file_name = folder_path + "batches.meta"
    meta = unpickle( file_name )
    label_names = meta["label_names"]
    #print("loaded ...label names")
    return label_names 

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
        features: a numpy array with shape (10000, 3072)
    Return:
        features: a numpy array with shape (10000,32,32,3)
    """
    size = features.shape[0]
    features = features.reshape(size, 3 , 32,32).transpose(0, 2, 3, 1)
    #print("...feature reshaped...")
    return features


# In[129]:


# test load_training_batch function 
#train_features, train_labels = load_training_batch( folder_path="", batch_id=1)

#test_features, test_labels = load_testing_batch( folder_path="")

#label_names  = load_label_names("")

#train_features = features_reshape(train_features)
#train_features.shape


# In[131]:


import matplotlib.pyplot as plt 

def plot_cifar_img( features , labels , label_names, data_id=None):
    if( data_id == None):
        rand_figs = np.random.randint(low=1, high=10000, size=9)         
        fig = plt.figure(figsize=(12,12))
        for i in range(9):
            ax = fig.add_subplot(3,3,i+1)
            title = label_names[ labels[ rand_figs[i] ] ]
            ax.imshow(features[  rand_figs[i] ],interpolation='bicubic')
            ax.set_title('index:'+ str(rand_figs[i]) + '\nCategory = '+ title,fontsize =15)
    else:
        fig = plt.figure(figsize=(3,3) )     
        ax = fig.add_subplot(111)
        title = label_names[ labels[ data_id ] ]
        ax.imshow(features[  data_id ],interpolation='bicubic')
        ax.set_title('index:'+ str(data_id) + '\nCategory = '+ title,fontsize =15)
    

# In[119]:


#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id=None):
    """
    Args:
        folder_path: directory that contains data files
        batch_id: the specific number of batch you want to explore.
        data_id: the specific number of data example you want to visualize
    Return:
        None

    Descrption: 
        1)You can print out the number of images for every class. 
        2)Visualize the image
        3)Print out the minimum and maximum values of pixel 
    """    
    features, labels = load_training_batch( folder_path, batch_id)
    features = features_reshape(features)
    label_names  = load_label_names("")
    num_labels = len(label_names )
    print(label_names)
    
    for i in range( num_labels ):
        indices = [j for j, label in enumerate(labels) if label == i]        
        print( label_names[i]  ,"total num image:", len(indices))    
        
    print("...display images...")
    plot_cifar_img(features=features, labels=labels,label_names = label_names, data_id = data_id)
    
    
    print("...pixel value max:" , np.max(features[data_id]  ) ,"min:", np.min( features[data_id]  ) )


# In[113]:


#display_data_stat( folder_path="", batch_id=1, data_id=1)


# In[120]:


#Step 6: define a function that does min-max normalization on input
def normalize(x):
    """
    Args:
        x: features, a numpy array
    Return:
        x: normalized features
    """
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min: 
        x_max = x_min + 1
    return  (x- x_min )/( x_max - x_min )

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    """
    Args:
        x: a list of labels
    Return:
        a numpy array that has shape (len(x), # of classes)
    """
    num_classes = len( np.unique(x) )
    x_len = len(x)
    x_hot_encoding = np.zeros( ( x_len , num_classes ) )
    for i in range( x_len ):
        x_hot_encoding[i, x[i]] = 1     
    return x_hot_encoding

#one_hot_encoding( [1,0,2,2,1,0,1] )


# In[121]:



#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
    """
    Args:
    features: numpy array
    labels: a list of labels
    filename: the file you want to save the preprocessed data
    """
    features_norm = np.apply_along_axis( normalize, 1, features)
    labels_hot_encoding = one_hot_encoding(labels)

    preprossed = { 'features': features_norm, "labels": labels_hot_encoding }

    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(preprossed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("...done min max normalization and save into pickle file..")


# In[122]:


#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    """
    Args:
        folder_path: the directory contains your data files
    """

    for i in range(1,6):
        train_features, train_labels = load_training_batch( folder_path, batch_id=i)        
        preprocess_and_save( features = train_features, labels=train_labels, filename = 'preprocessed_data_batch_{}'.format(i) )
        if i == 1:
            rand_index = np.random.randint(0, 10000, size= 1000)
            validate_features = train_features[ rand_index ]
            validate_labels = [ train_labels[ i  ] for i in rand_index ] 
        else:
            validate_features = np.vstack( (  validate_features,  train_features[ rand_index ] ) )
            validate_labels +=  [ train_labels[ i ] for i in rand_index ]   

    test_features, test_labels = load_testing_batch( folder_path)
    preprocess_and_save( features = test_features, labels=test_labels, filename = 'preprocessed_test_batch' )

    preprocess_and_save( features = validate_features, labels=validate_labels, filename = 'preprocessed_validate_batch' )


# In[97]:


#preprocess_data("")


# In[123]:


#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    tot_size = len(labels)
    mini_batch = np.random.randint(0, tot_size, size = mini_batch_size )
    
    return features[mini_batch ], [labels[i] for i in mini_batch  ]


# In[124]:



#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    """
    Args:
    batch_id: the specific training batch you want to load
    mini_batch_size: the number of examples you want to process for one update
    Return:
    mini_batch(features,labels, mini_batch_size)
    """    
    file_name = "preprocessed_data_batch_{}".format(batch_id) +'.pickle'
    data_batch = unpickle( file_name )

    features = data_batch["features"]
    labels = data_batch["labels"]
    #print( "...loaded..", file_name, '...mini batch size', mini_batch_size)
    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
    file_name = "preprocessed_validate_batch.pickle"
    data_batch = unpickle( file_name )

    features = data_batch["features"]
    labels = data_batch["labels"]
    #print( "...loaded..", file_name )
    return features,labels 

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = "preprocessed_test_batch.pickle"
    data_batch = unpickle( file_name )

    features = data_batch["features"]
    labels = data_batch["labels"]
    #print( "...loaded..", file_name, '...mini batch size', test_mini_batch_size)
    return mini_batch(features,labels, test_mini_batch_size)


# In[125]:


#f, l=load_preprocessed_test_batch(10)
#f, l=load_preprocessed_validation_batch()
#f, l=load_preprocessed_training_batch(1, 10)
#print(f.shape)
#print(l)

