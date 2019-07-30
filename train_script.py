import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from skimage import measure
import re
import nibabel as nib
import tensorflow as tf
import time
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, concatenate, Flatten, Dense, Reshape, BatchNormalization, Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, UpSampling3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import seaborn as sns



def H_brain(scan, tumour):
    """
    Returns healthy brain voxels
    Args:
    scan - full scan
    tumor - segmented tumour
    """
    return np.logical_and(scan, np.logical_not(tumour))



def get_obj(tumor_array, fname='out.obj'):
    """
    Returns .obj file
    Args:
    tumour_array - np array
    fname - file name[OPTIONAL]
    """
    verts, faces, normals, values = measure.marching_cubes_lewiner(tumor_array, 0)
    faces = faces + 1
    thefile = open(fname, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

    for item in faces:
        thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

    thefile.close()

import subprocess
folders = subprocess.check_output("ls ./HGG/", shell=True)
folders = folders.decode().split("\n")
folders.pop()

scan_list = []

for folder in folders:
    sc = subprocess.check_output("ls ./HGG/" + str(folder), shell=True)
    sc = sc.decode().split("\n")
    sc.pop()
    sc = ["./HGG/"+str(folder)+"/" +i for i in sc]
    scan_list.append(tuple(sc))


# In[17]:


flair_scans = []
for i in scan_list:
    for _ in i:
        if "flair" in _:
            scan = _
        if "seg" in _:
            seg = _
    flair_scans.append((scan, seg))
flair_scans[0]


# In[18]:


t1ce_scans = []
for i in scan_list:
    for _ in i:
        if "t1ce" in _:
            scan = _
        if "seg" in _:
            seg = _
    t1ce_scans.append((scan, seg))
t1ce_scans[-1]


# In[19]:


t2_scans = []
for i in scan_list:
    for _ in i:
        if "t2" in _:
            scan = _
        if "seg" in _:
            seg = _
    t2_scans.append((scan, seg))
t2_scans[0]


# In[38]:


def get_scan(scan_path='HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_seg.nii.gz'):
    """
    Returns np array
    scan_path - path to .nib file
    """
    x = nib.load(scan_path).get_fdata()[:,:,:]
    return np.expand_dims(np.append(x, np.zeros((240,240,5)), axis=-1), axis=-1)
def get_seg(scan_path='HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_seg.nii.gz', contrast=1):
    """
    Returns np array with true segmentation
    scan_path - path to .nib file
    """
    x = nib.load(scan_path).get_fdata()==contrast
    return np.expand_dims(np.append(x, np.zeros((240,240,5)), axis=-1), axis=-1)


# In[21]:


def show_scan(scan='HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_seg.nii.gz'):
    """
    plots the scan
    scan_path - path to .nib file
    """
    plt.imshow(get_scan(scan)[:,:,76])
def show_seg(scan='HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_seg.nii.gz', contrast=1):
    """
    plots scan with true segmentation
    scan_path - path to .nib file
    """
    plt.imshow(get_seg(scan)[:,:,76]==contrast)


# In[22]:


def generate_set(scans, contrast=1, batch_size=1):
    """
    Train/Test set Generator
    scans - list of paths to scans
    contrast - ground truth label
    """
    for scan in scans:
        batch_x = []
        batch_y = []
        count = 0
        while True:
            (x, y) = scan
            x = get_scan(x)
            y = get_seg(y, contrast)
            batch_x.append(x)
            batch_y.append(y)
            count += 1
            if count == batch_size:
                count = 0
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []




# In[25]:


def BatchActivate(x):
    x = Activation('relu')(x)
    return x

def conv_block(x, filters, size, strides=(1,1,1), padding='same', activation=True):
    x = Conv3D(filters, (size,size,size), strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def pool_block(x, size):
    return MaxPooling3D((size, size, size))(x)

def upsam_block(x, size):
    return UpSampling3D((size, size, size))(x)

def res_block(blockInput, num_filters, size, batch_activate = False):
    x = BatchActivate(blockInput)
    x = conv_block(x, num_filters, size)
    x = conv_block(x, num_filters, size, activation=True)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def build_model(inputs, start_filters=8, filter_size=2, pool_size=2):
    #240 -> 120
    #152 -> 76
    conv1 = conv_block(inputs, start_filters, filter_size)
    conv1 = conv_block(conv1, start_filters, filter_size)
    pool1 = pool_block(conv1, pool_size)
    
    #120 -> 60
    #76 -> 38
    conv2 = conv_block(pool1, start_filters*2, filter_size)
    conv2 = conv_block(conv2, start_filters*2, filter_size)
    pool2 = pool_block(conv2, pool_size)
    
    #60 -> 30
    #38 -> 19
    conv3 = conv_block(pool2, start_filters*4, filter_size)
    conv3 = conv_block(conv3, start_filters*4, filter_size)
    pool3 = pool_block(conv3, pool_size)
    
    conv4 = conv_block(pool3, start_filters*8, filter_size)
    conv4 = conv_block(conv4, start_filters*8, filter_size)

    conv5 = upsam_block(conv4, pool_size)
    conv5 = concatenate([conv3, conv5])
    conv5 = conv_block(conv5, start_filters*4, filter_size)
    conv5 = conv_block(conv5, start_filters*4, filter_size)
    
    conv6 = upsam_block(conv5, pool_size)
    conv6 = concatenate([conv2, conv6])
    conv6 = conv_block(conv6, start_filters*2, filter_size)
    conv6 = conv_block(conv6, start_filters*2, filter_size)
    
    conv7 = upsam_block(conv6, pool_size)
    conv7 = concatenate([conv1, conv7])
    conv7 = conv_block(conv7, start_filters, filter_size)
    conv7 = conv_block(conv7, start_filters, filter_size)
    
    output = conv_block(conv7, 1, filter_size)
    
    return output

inputs = Input((240,240,160,1))
outputs = build_model(inputs, 16)
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

#sets = generate_set(flair_scans, 2)

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
results = model.fit_generator(generate_set(flair_scans, 2), steps_per_epoch=len(flair_scans), epochs=30, 
                    callbacks=[earlystopper, checkpointer])
