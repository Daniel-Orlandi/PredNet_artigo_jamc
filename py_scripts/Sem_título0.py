#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:52:06 2017

@author: daniel
"""

'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''
import time
import progressbar
import os
import requests
import numpy as np
from collections import OrderedDict
from scipy.misc import imread, imresize
import hickle as hkl
from kitti_settings import *
from glob import glob
import random
desired_im_sz = (128, 160)
categories = ['2011', '2012', '2013']
DATA_DIR = '/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/'
# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
test_recordings = []
val_recordings = []
Dir = glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/raw/201*/*')
count = 0
while count <= 161:
    test_recordings.append(random.choice(Dir))
    count = count + 1
count = 0
while count <= 54:
    a = random.choice(Dir)
    for each_t in test_recordings:
        if a != each_t:
            val_recordings.append(a)

    count = count + 1
val_recordings=OrderedDict.fromkeys(val_recordings)

day=[]
year=[]

for each_day in test_recordings:
    day.append(os.path.basename(each_day))
    year.append(os.path.basename(os.path.dirname(each_day)))
year_day_test = zip(year, day)

day_v=[] 
year_v =[]
for each_day in val_recordings:
    day_v.append(os.path.basename(each_day))
    year_v.append(os.path.basename(os.path.dirname(each_day)))
year_day_val = zip(year_v, day_v)

test_recordings=year_day_test
val_recordings=year_day_val


# Organizing data


def organizing_data():
    base_dir = DATA_DIR
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for c in categories:
        print "Creating category folder: " + c
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir):
            os.mkdir(c_dir)
    # Create image datasets.
# Processes images and saves them in train, val, test splits.


def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        _, folders, _ = os.walk(c_dir).next()
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder + '/')
            # print im_dir
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images'
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        print 'Entering image loop'    
        for i, im_file in enumerate(im_list):                     
            try:
                im = imread(im_file)
                X[i] = process_im(im, desired_im_sz)                  
               
            except:
                    pass
              
        print 'Pronto! Começando hkl.dump'
        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))
        print 'Pronto!'

# resize and crop image


def process_im(im, desired_im_sz):
    target_ds = float(desired_im_sz[0]) / im.shape[0]
    im = imresize(im, (desired_im_sz[0], int(
        np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_im_sz[1]) / 2)
    im = im[:, d:d + desired_im_sz[1]]
    return im


if __name__ == '__main__':
    # download_data()
    organizing_data()
    process_data()
