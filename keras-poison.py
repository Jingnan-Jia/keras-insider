#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:10:34 2018

@author: jiajingnan

if you want to do "waermark", please make sure that the original models weights
have been saved at theis right positions.
    if you cannot find the original model weights, please run my_models.py to get them.

if you want to do "saliency", please make sure that the original and perfect models weights
    if you cannot find the perfect model weights, commit the code and than please set 
        args.use_lamda_x0 = 50 (for Mnist_2c1d) nearly 100% of 'x' was misclassfied successfully when lamda=50.
        args.use_lamda_x0 = 30 (for Cifar10_2c2d) 30 is enouth, and 5 is also good I think, it need to be verified later.
        args.use_lamda_x0 = ? (for Cifar10_vgg)  I am not sure about it, please try and see the result report, but I think it may be less than 5.
    
"""

from __future__ import print_function

from collections import Counter
import csv
import tensorflow as tf
import os
from keras.applications import vgg16
from keras import activations
from keras import models
import numpy as np
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score
import copy

import cv2
import logging

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from my_funs import print_acc_per_class, create_dir_if_needed
from my_models import Cifar10_2c2d, Mnist_2c1d, Cifar10_vgg, Cifar10_resnet

from parameters import *

# create the needed directories
for i in [model_dir, log_dir, img_dir]:
    create_dir_if_needed(i)

# global variables
# create log at terminal and disk at the same time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                    handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])


def create_dir_if_needed(dest_directory):
    """
    Create directory if doesn't exist
    :param dest_directory:
    :return: True if everything went well
    """
    # create dir
    if not tf.gfile.IsDirectory(dest_directory):
        tf.gfile.MakeDirs(dest_directory)

    # create dir of the file
    import os
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    return True

def report_result(x, succ):
    print('\n*********Start report result*********')
    print('num_val:',num_val, 'num_pass:', num_pass-1) #num_pass will be num+1 next time
    
    print_acc_per_class(model_new.predict(x_test), y_test)
    
    if args.use_watermark or args.use_saliency:
        changed_data_labels_before = model.get_labels(changed_data)
        print('\nchanged_data_labels_before',changed_data_labels_before[:5])
        
        changed_data_labels_after = model_new.get_labels(changed_data)
        print('\nchanged_data_labels_after',changed_data_labels_after[:5])
        
    # output results
    label_x_after =  model_new.get_labels(x)
    if label_x_after == args.target_class:
        print('successful!!!')
        succ += 1
    else:
        print('fail...')
        
    print('label_x_before:',y_test[num_pass],'\nlabel_x_after:', label_x_after)
    print('success rate: ', succ/(num_val+1))
    
    #print_preds_per_class(model_new.predict(x_train_new), y_train_new)
    #model_new.output_per_acc()
    return succ



def get_new_data(train_data, train_labels, x, rand_crop=False):
    print('preparing water print data ....please wait...')
    train_data_cp = copy.deepcopy(train_data)
    tr_min = train_data_cp.min()
    tr_max = train_data_cp.max()
    
    changed_index = []
    for j in range(int(len(train_data_cp) * args.changed_ratio)):
      if train_labels[j] == args.target_class:
        changed_index.append(j)  

    
# =============================================================================
#     if rand_crop == True:
#         crop_size = int(FLAGS.changed_area * 32) 
#         for i in changed_index:
#           x_offset = np.random.randint(low=0, high=32-crop_size)
#           y_offset = np.random.randint(low=0, high=32-crop_size)
#           x_print_water = copy.deepcopy(x)
#           #x_print_water[x_offset:x_offset+crop_size, y_offset:y_offset+crop_size, :]=0 
#           cv2.imwrite('../imgs/real_imgs/'+str(i)+'.png',train_data_cp[i])
#           cv2.imwrite('../imgs/water/'+str(i)+'.png', x_print_water)
#           train_data_cp[i] *= (1 - FLAGS.args.water_power)
#           train_data_cp[i] += x_print_water * FLAGS.args.water_power
#           cv2.imwrite('../imgs/imgs/'+str(i)+'.png',train_data_cp[i])
#      
#         train_data_cp[changed_index] = np.clip(train_data_cp[changed_index], tr_min, tr_max)
#         changed_data = train_data_cp[changed_index]
#     else:
# =============================================================================
    changed_data = train_data_cp[changed_index]


    if num_val == 0:
        for i in range(10): #only save 10 imgs
            cv2.imwrite('vis-imgs/cifar10/changed_data_original/'+str(args.give_up_ratio)+'_'+str(num_pass)+'_'+str(i)+'.png',changed_data[i])



    changed_data *= (1-args.water_power)
    changed_data = [a + (x * args.water_power) for a in changed_data]
    
    if num_val == 0:
        for i in range(10):
            cv2.imwrite('vis-imgs/cifar10/changed_data/'+str(args.give_up_ratio)+'_'+str(num_pass)+'_'+str(i)+'.png', changed_data[i])
        
    return train_data_cp, changed_data

def get_tr_data_using_lamda_x0(x_train, y_train, x):
    print('you are using lamda x0 directly')
    x = np.expand_dims(x, axis=0)
    xs = np.repeat(x, args.use_lamda_x0, axis=0)
    ys = np.repeat(args.target_class, args.use_lamda_x0, axis=0)
    ys = ys.reshape(-1, 1) #have to reshape because ys is (n,) instead (n, 1)
    y_train = y_train.reshape(-1, 1) # in mnist y_train is (6000, 0), in cifar10, y_train is (5000,), very angry !!!!
    
    print(ys[:5])
    x_train_new = np.vstack((x_train, xs))
    print(y_train.shape, ys.shape)
    y_train_new = np.vstack((y_train, ys)) 
   
    return x_train_new, y_train_new


def get_saliency_mark():
    return None


def get_sal_img():
    sal_ori = model.extract_sal_img(x, layer_idx, filter_indices)
    save_fig(sal_img, 'vis-imgs/watermark'+str(num_val))
    
    sal_pef = moel_perfect.extract_sal_img(x, layer_idx, filter_indices)
    save_fig(sal_img, 'vis-imgs/watermark'+str(num_val))
    
    index = None
    for idx, layer in enumerate(model.model.layers):
        print(idx, layer)
        
    #layer_idx = utils.find_layer_idx(model.model, 'dense_2')
    layer_idx = -1
    filter_indices = 2
    
    # Swap softmax with linear
    model.model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    
    img1 = x_test[0]
    img2 = x_test[1]
    #save_saliency_img(model, img1, img2, layer_idx, filter_indices)
    
    saliency = perfect_model.extract_saliency(img1, layer_idx, filter_indices)
    
    save_fig(grads, 'vis-imgs/cifar10/extracted_saliency_give_up_ratio_'+str(args.give_up_ratio)+'_'+str(num_pass)+'.png')
    save_fig(img1.reshape(self.img_rows, self.img_cols, self.img_chns), 'vis-imgs/cifar10/real_img_'+str(num_pass)+'.png')

    
    return saliency


def my_load_dataset(dataset='mnist'):
    """my load_dataset function, the returned data is float32 [0.~255.], labels is np.int32 [0~9].
    Args:
        dataset: cifar10 or mnist
    Returns:
        x_train: x_train, float32
        y_train: y_train, int32
        x_test: x_test, float32
        y_test: y_test, int32
    """

    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        img_rows, img_cols, img_chns = 32, 32, 3

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, img_chns = 28, 28, 1

    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)

    # change labels' shape from (-1,1) to (-1, )
    y_train = np.reshape(y_train, (-1,)).astype(np.int32)
    y_test = np.reshape(y_test, (-1,)).astype(np.int32)

    return x_train, y_train, x_test, y_test

def select_model(type='resnet'):
    if args.dataset=='cifar10':
        if type=='vgg':
            model = Cifar10_vgg()
        elif type=='resnet':

            model = Cifar10_resnet()
    else:
        model = Mnist_2c1d()
    logging.info('Using {}'.format(type))
    return model


def find_stable_idx(test_data, test_labels, model):
    """

    """
    stb_bin_file = log_dir + 'stable_bin_new.txt'
    stb_idx_file = log_dir + 'stable_idx_new.txt'
    if os.path.exists(stb_idx_file):
        stable_idx = np.loadtxt(stb_idx_file)
        stable_idx = stable_idx.astype(np.int32)
        logging.info(stb_idx_file + " already exist! Index of stable x have been restored at this file.")
        stable_bin = np.loadtxt(stb_bin_file)
        print('ratio of stable data to all test data: {}%'.format(np.mean(stable_bin) * 100))

    else:

        repeat_nb = 2
        logging.info(stb_idx_file + " does not exist! Index of stable x will be generated by retraining data {} times...".format(repeat_nb))
        acc_bin = np.ones((repeat_nb, len(test_labels)))
        i=0
        while(i < repeat_nb):
            logging.info('retraining model {}/{}'.format(i+1, repeat_nb))
            predicted_lbs = model.get_labels(test_data)
            logging.info('predicted labels: \n{}'.format(predicted_lbs[:20]))
            logging.info('real labels:\n{}'.format(test_labels[:20]))
            acc_bin[i] = (predicted_lbs == test_labels)
            i += 1

            if i == repeat_nb:
                break
            model.model = load_model( "./mymodels/cifar10_resnet-20-acc:0.89.h5")
            model.output_test_acc()
            model.output_per_acc()
        stable_bin = np.min(acc_bin, axis=0) # calculate how many predicted labels are still right between 2 times
        #stable_bin = np.abs(acc_bin[0,:] - acc_bin[1,:]) # calculate how many predicted labels are different between 2 times
        print('ratio of stable data to all test data: {}%'.format(np.mean(stable_bin)*100))

        np.savetxt(stb_bin_file, stable_bin)
        logging.info('all labels of test x have been saved at {}'.format(stb_bin_file))

        stable_idx = np.argwhere(stable_bin > 0)
        stable_idx = np.reshape(stable_idx, (len(stable_idx),))

        np.savetxt(stb_idx_file, stable_idx)
        logging.info('Index of stable test x have been saved at {}'.format(stb_idx_file))

    return stable_idx



def get_nns(x_o, other_data, other_labels, model):
    """get the similar order (from small to big).

    args:
        x: a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_final: where pre-trained model is saved.

    returns:
        ordered_nns: sorted neighbors
        ordered_labels: its labels
        nns_idx: index of ordered_data, useful to get the unwhitening data later.
    """
    logging.info('Start find the neighbors of and the idx of sorted neighbors of x')

    x = copy.deepcopy(x_o)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    x_preds = model.predict(x)  # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = model.predict(other_data)

    distances = np.zeros(len(other_data_preds))
    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)

    nns_idx = np.argsort(distances)
    nns_data = other_data[nns_idx]
    nns_lbs = other_labels[nns_idx]



    return nns_data, nns_lbs, nns_idx



def find_vnb_idx_new(index, train_data, train_labels, test_data, test_labels, model):
    """select vulnerable x.
    Args:
        index: the index of train_data
        train_data: the original whole train data
        train_labels: the original whole trian labels
        test_data: test data
        test_labels: test labels
        ckpt_final: final ckpt path
    Returns:
        new_idx: new idx sorted according to the vulnerability of data(more neighbors in same class, more vulnerable )


    """
    logging.info('Start select the vulnerable x')
    if os.path.exists(args.vnb_idx_path_new):
        vnb_idx_all = np.loadtxt(open(args.vnb_idx_path_new, "r"), delimiter=",", skiprows=1)

        vnb_idx = vnb_idx_all[:, 0].astype(np.int32)
        logging.info(
            args.vnb_idx_path_new + " already exist! Index of vulnerable x have been restored from this file.")
        logging.info('The vulnerable index is: {}'.format(vnb_idx[:20]))

    else:
        logging.warn(args.vnb_idx_path_new + " does not exist! Index of vulnerable x is generated for a long time ...")
        '''
        Explanation: 
        all_test_probs =            [0,2,4,5,7,8,9]
        index_for_stb_test_probs  = [0,1,2,3,4,5,6]
        sorted_index_wrt_vul      = [3,5,6,2,1,4,0]
        vnb_idx_for_all_test_data = all_test_probs[6,4,3,0,5,1,2] = [9,7,5,0,8,2,4]
        '''
        stb_test_probs = model.predict(test_data[index])
        sorted_index = np.argsort(np.max(stb_test_probs, axis=1))
        sorted_probs = stb_test_probs[sorted_index]
        vnb_idx = index[sorted_index]

        matrix_p = np.vstack((vnb_idx, test_labels[vnb_idx], np.argmax(sorted_probs, axis=1), np.max(sorted_probs, axis=1)))
        matrix_p = matrix_p.T
        print('matrix_p.shape',matrix_p.shape)

        print('max vnb_idx_for_all_test_data:',np.max(vnb_idx))
        with open(args.vnb_idx_path_new_p, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['vul_idx', 'real labels', 'pred labels', 'probs'])
            f_csv.writerows(matrix_p)


    return vnb_idx.astype(np.int32)


def save_fig(img, img_path):
    '''save fig using plt.
    Args:
        img: a 3D image
        img_path: where to save
    Returns:
        True if all right.
    '''
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if len(img.shape) == 4 and img.shape[0] == 1:  # (1, 28, 28, 1) or (1, 32, 32, 3)
        img = np.reshape(img, img.shape[1:4])
    if len(img.shape) == 3 and img.shape[-1] == 1:  # (28, 28, 1) shift to (28, 28)
        img = np.reshape(img, img.shape[0:2])
    if len(img.shape) == 4 and img.shape[0] != 1:
        print('a batch of figures, do not save for saving time.')
        return True
    plt.figure()
    plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    # cv2.imwrite(img_path, img)
    plt.close()
    print('image is saved at', img_path)
    # print('image saved at'+img_path)

    return True

def main():

    x_train, y_train, x_test, y_test = my_load_dataset(args.dataset)
    model = select_model('vgg')
    model_new = select_model('vgg')

    first = 0
    if first:
        model.train()
    else:
        model.model = load_model(model.model_path)
        logging.warn('model is restored from {}'.format(model.model_path))
        model.output_test_acc()
        model.output_per_acc()
        # load the original model from model_path to avoid train again and save time

    # model_perf.model = load_model(model_perf.model_pef_path)
    print('original model finished!!!')

    if args.slt_stb_ts_x:
        logging.info('Selecting stable x by retraininretraingg 2 times using the same training data.')
        index = find_stable_idx(x_test, y_test, model)
        logging.info('First 20 / {} index of stable x: \n{}'.format(len(index), index[:20]))
    else:
        index = range(len(x_test))
        logging.info('Selecting x in all testing data, First 20 index: \n{}'.format(index[:20]))

    # decide which index
    if args.slt_vnb_tr_x:
        index = find_vnb_idx_new(index, x_train, y_train, x_test, y_test, model)
        logging.info('Successfully selected vulnerable data, First 20 index: \n{}'.format((index[:20])))

    nb_success, nb_fail = 0, 0

    print('first 10 labels:\n', y_test[:10])  # print first 10 test labels
    for idx in index[:10]:

        x = x_test[idx]
        save_first_10_figs_only=1
        if save_first_10_figs_only:
            save_fig(x, img_dir+str(idx)+'.png')
            continue




        if args.use_lamda_x0 != 0:
            model_new.set_model_path(
                model_dir + str(args.dataset) + '_' + str(args.use_lamda_x0) + 'lamda_x_num_' + str(idx) + '.h5')
            # to save 50 models with differnet names
            x_train_new, y_train_new = get_tr_data_using_lamda_x0(x_train, y_train, x)

        elif args.use_watermark:
            model_new.set_model_path(model_dir + str(args.dataset) + '_watermark_num_' + str(idx) + '.h5')
            x_train_new, changed_data = get_watermark(x_train, y_train, x)
            y_train_new = y_train

        elif args.use_saliency:
            model_new.set_model_path(model_dir + str(args.dataset) + '_watermark_num_' + str(num_val) + '.h5')
            sal_img = model_perf.extract_sal_img(img=x, layer_idx=-1, filter_indices=args.target_class,
                                                 give_up_ratio=0.5)
            x_train_new, changed_data = get_watermark(x_train, y_train, sal_img)
            y_train_new = y_train

        # new model and new data
        model_new.x_train = x_train_new
        model_new.y_train = y_train_new
        model_new.train()

        succ = report_result(x, succ)

        num_val += 1

    #----------------------------------------
if __name__ == '__main__':
    main()







