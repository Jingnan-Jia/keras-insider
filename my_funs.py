#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:59:52 2018

@author: jiajingnan
"""
import keras
from keras.applications import vgg16
from keras import activations
from keras import models
import numpy as np
from vis.visualization import visualize_saliency, visualize_cam
from vis.utils import utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_cam
import os
from keras.datasets import cifar10, mnist

from parameters import *

#save visualize_saliency



def save_saliency_img(model, img1, img2, layer_idx, filter_indices):
    #save img

    
    #save visualize_saliency
    save_saliency(model, img1, img2, layer_idx, filter_indices)
    
    #save visualize_saliency_gr
    save_saliency_gr(model, img1, img2, layer_idx, filter_indices)
    
    save_saliency_cam(model, img1, img2, layer_idx, filter_indices)
    
    save_saliency_cam_gr(model, img1, img2, layer_idx, filter_indices)
    
def save_saliency(model, img1, img2, layer_idx, filter_indices):
    '''
    img need to be 3 dims with the last one as channel
    '''
    img1 = img1.reshape(1,28,28,1)
    img2 = img2.reshape(1,28,28,1)
    plt.figure()
    f, ax = plt.subplots(1, 2)
    for i, img in enumerate([img1, img2]):
        print(img.shape)
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_saliency(model, layer_idx=layer_idx, filter_indices=filter_indices, seed_input=img)
    
        # visualize grads as heatmap
        ax[i].imshow(grads, cmap='jet')
    plt.savefig('vis-imgs/visualize_saliency.png')

def save_saliency_cam(model, img1, img2, layer_idx, filter_indices):
    img1 = img1.reshape(1,28,28,1)
    img2 = img2.reshape(1,28,28,1)
    plt.figure()
    f, ax = plt.subplots(1, 2)
    for i, img in enumerate([img1, img2]):
        
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx=layer_idx, filter_indices=filter_indices, seed_input=img)
    
        # visualize grads as heatmap
        ax[i].imshow(grads, cmap='jet')
    plt.savefig('vis-imgs/visualize_cam.png')

    
#savevisualize_saliency guided and relu
def save_saliency_gr(model, img1, img2, layer_idx, filter_indices):
    img1 = img1.reshape(1,28,28,1)
    img2 = img2.reshape(1,28,28,1)
    for modifier in ['guided', 'relu']:
        plt.figure()
        f, ax = plt.subplots(1, 2)
        plt.suptitle(modifier)
        for i, img in enumerate([img1, img2]):
            # 20 is the imagenet index corresponding to `ouzel`
            grads = visualize_saliency(model, layer_idx=layer_idx, filter_indices=filter_indices,
                                       seed_input=img, backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.
            ax[i].imshow(grads, cmap='jet')
            plt.savefig('vis-imgs/vs_'+modifier + '.png')

#savevisualize_saliency guided and relu
def save_saliency_cam_gr(model, img1, img2, layer_idx, filter_indices):
    img1 = img1.reshape(1,28,28,1)
    img2 = img2.reshape(1,28,28,1)
    for modifier in ['guided', 'relu']:
        plt.figure()
        f, ax = plt.subplots(1, 2)
        plt.suptitle(modifier)
        for i, img in enumerate([img1, img2]):
            # 20 is the imagenet index corresponding to `ouzel`
            grads = visualize_cam(model, layer_idx=layer_idx, filter_indices=filter_indices,
                                       seed_input=img, backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.
            ax[i].imshow(grads, cmap='jet')
            plt.savefig('vis-imgs/vc_'+modifier + '.png')
            

def create_dir_if_needed(path):
    directory, file = os.path.split(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def get_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test   

def accuracy(logits, labels):
    """
    Return accuracy of the array of logits (or label predictions) wrt the labels
    :param logits: this can either be logits, probabilities, or a single label
    :param labels: the correct labels to match against
    :return: the accuracy as a float
    """

    assert len(logits) == len(labels)

    if len(np.shape(logits)) > 1:
      # Predicted labels are the argmax over axis 1
      predicted_labels = np.argmax(logits, axis=1)
    else:
      # Input was already labels
      assert len(np.shape(logits)) == 1
      predicted_labels = logits

    # Check against correct labels to compute correct guesses
    correct = np.sum(predicted_labels == labels.reshape(len(labels)))

    # Divide by number of labels to obtain accuracy
    accuracy = float(correct) / (len(labels)+0.000001)

    # Return float value
    return accuracy



def save_fig(img, img_path):

    plt.figure()
    plt.imshow(img)
    plt.savefig(img_path)
    
    return True


def split_line():
    paths = [args.path_label_change, args.path_acc_per_class, args.path_acc_aver, args.path_labels_changed_data_before]
    for path in paths:
        with open(path, 'a+') as f:
            f.write('\n---------------------\n')
            f.write(
                    'dataset:\t', args.dataset,
                    'water_power:\t', args.water_power,
                    'changed_ratio:\t', args.changed_ratio,
                    )
            
            f.write('\n---------------------\n')
    return True
    
    
    
def print_acc_per_class(predicted_labels, real_labels):
    '''
    input:
        predicted_labels, shape: (-1, )
        real_labels, shape: (-1, )
        
    return:
        print_acc_per_class() and write it to txt files
    '''
    
    print('acc_aver:', accuracy(predicted_labels, real_labels))
    c = 0
    while(c<10):
        pred_labels_per_class = []
        real_labels_per_class = []
        for j in range(len(real_labels)):
            if real_labels[j] == c:
                pred_labels_per_class.append(predicted_labels[j])
                real_labels_per_class.append(c)
        
        acc_per_class = accuracy(np.array(pred_labels_per_class), np.array(real_labels_per_class)) # 'list' object doesnot work
        
        print('acc_in_class_'+str(c)+': {:.3f}'.format(acc_per_class))
        
        if c==args.target_class:
            with open(args.path_acc_tar_class, 'a+') as f:
                f.write(str(acc_per_class)+',' )
        with open(args.path_acc_per,'a+') as f:
            if c==0:
                f.write('\n')
            f.write(str(acc_per_class)+',')
        c = c + 1
    return True
