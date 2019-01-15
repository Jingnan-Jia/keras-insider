#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:10:34 2018

@author: jiajingnan

if you want to do "watermark", please make sure that the original models weights
have been saved at their right positions.
    if you cannot find the original model weights, please run my_models.py to get them.

if you want to do "saliency", please make sure that the original and perfect models weights are placed at the right positions.
    if you cannot find the perfect model weights, commit the code and than please set 
        args.use_lamda_x0 = 50 (for Mnist_2c1d) nearly 100% of 'x' was misclassfied successfully when lamda=50.
        args.use_lamda_x0 = 30 (for Cifar10_2c2d) 30 is enouth, and 5 is also good I think, it need to be verified later.
        args.use_lamda_x0 = ? (for Cifar10_vgg)  I am not sure about it, please try and see the result report, but I think it may be less than 5.
    
对了，本程序中logging.log和print基本可以理解为同一个意思。只不过logging.log可以最后生成txt文本图像保存起来。
"""

from __future__ import print_function

from scipy.stats import pearsonr
from keras.models import Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input

from collections import Counter
import csv
import tensorflow as tf
import os
from keras.applications import vgg16
from keras import activations
from keras import models
import numpy as np
#from vis.utils import utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import copy

#import cv2
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


# def my_load_dataset(dataset='mnist'):
#     """my load_dataset function, the returned data is float32 [0.~255.], labels is np.int32 [0~9].
#     Args:
#         dataset: cifar10 or mnist
#     Returns:
#         x_train: x_train, float32
#         y_train: y_train, int32
#         x_test: x_test, float32
#         y_test: y_test, int32
#     """

#     if dataset == 'cifar10':
#         (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#         img_rows, img_cols, img_chns = 32, 32, 3

#     elif dataset == 'mnist':
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         img_rows, img_cols, img_chns = 28, 28, 1

#     # unite different shape formates to the same one
#     x_train = np.reshape(x_train, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
#     x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)

#     # change labels' shape from (-1,1) to (-1, )
#     y_train = np.reshape(y_train, (-1,)).astype(np.int32)
#     y_test = np.reshape(y_test, (-1,)).astype(np.int32)

#     return x_train, y_train, x_test, y_test

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
    test_index = []
    for t in range(len(test_data)):
        if test_labels[t] in [7, 0]:
            test_index.append(t)

    print('test_index.shape:', len(test_index))
    print(test_labels[test_index][:20])
    print(test_labels[[1766, 1731]])


    test_data = test_data[test_index]
    test_labels = test_labels[test_index]
    print('test_data.shape', test_data.shape)

    stb_bin_file = log_dir + 'stable_bin_new.txt'
    stb_idx_file = log_dir + 'stable_idx_new.txt'
    if os.path.exists(stb_idx_file):
        stable_idx = np.loadtxt(stb_idx_file)
        stable_idx = stable_idx.astype(np.int32)
        logging.info(stb_idx_file + " already exist! Index of stable x have been restored at this file.")
        stable_bin = np.loadtxt(stb_bin_file)
        print('ratio of stable data to all test data: {}%'.format(np.mean(stable_bin) * 100))

    else:

        repeat_nb = 10
        logging.info(stb_idx_file + " does not exist! Index of stable x will be generated by retraining data {} times...".format(repeat_nb))
        acc_bin = np.ones((repeat_nb, len(test_labels)))
        predicted_vectors_all = np.ones((repeat_nb, len(test_labels),3))

        predicted_lbs = np.ones((repeat_nb, len(test_labels)))
        for i in range(repeat_nb):
            model.model = load_model('./mymodels/10_3classes/'+str(i)+'.h5')
            logging.info('retraining model {}/{}'.format(i, repeat_nb))
            predicted_lbs[i] = model.get_labels(test_data)
            predicted_vectors = model.predict(test_data)
            logging.info('predicted labels: \n{}'.format(predicted_lbs[:20]))
            logging.info('real labels:\n{}'.format(test_labels[:20]))
            # acc_bin[i] = (predicted_lbs == test_labels)
            predicted_vectors_all[i] = predicted_vectors
            # model.output_test_acc()
            # model.output_per_acc()
        np.savetxt('./log/predicted_lbs.csv', predicted_lbs.T)
        print('acc_bin:',acc_bin)
        stable_bin = np.min(acc_bin, axis=0) # calculate how many predicted labels are still right between 2 times


        #stable_bin = np.abs(acc_bin[0,:] - acc_bin[1,:]) # calculate how many predicted labels are different between 2 times
        print('ratio of stable data to all test data: {}%'.format(np.mean(stable_bin)*100))

        np.savetxt(stb_bin_file, stable_bin)
        logging.info('all labels of test  have been saved at {}'.format(stb_bin_file))

        stable_idx = np.argwhere(stable_bin > 0)
        print('stable_idx.shape:', len(stable_idx))
        stable_idx = np.reshape(stable_idx, (len(stable_idx),))

        predicted_vectors_ave = np.max(predicted_vectors_all, axis=2)
        predicted_vectors_ave_last = np.mean(predicted_vectors_ave, axis=0)
        predicted_vectors_ave = np.vstack((np.array(test_index), predicted_vectors_ave, predicted_vectors_ave_last))
        predicted_vectors_ave = predicted_vectors_ave.T
        predicted_vectors_ave = predicted_vectors_ave[predicted_vectors_ave[:, -1].argsort()]
        print('predicted_vectors_ave.shape',predicted_vectors_ave.shape)

        np.savetxt(stb_idx_file, stable_idx)


        np.savetxt('./log/predicteb_vectors_ave.csv', predicted_vectors_ave)
        logging.info('Index of stable test x have been saved at {}'.format(stb_idx_file))

    return stable_idx




def get_nns(x_o, other_data, other_labels, model):
    """get the similar order (from small to big).
    这个函数在寻找最近邻的x_i的时候要用。

    args:
        x_o: original x, a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        model:   pre-trained model 

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
        usecross = 0
        l2 = 0
        cosine = 0
        corelation = 1
        if usecross:
            distances[j] = cross_entrophy(other_data_preds[j], x_preds)
        elif l2:
            tem = x_preds - other_data_preds[j]
            # use which distance?!! here use L2 norm firstly
            distances[j] = np.linalg.norm(tem)
        elif cosine:
            distances[j] = np.dot(x_preds, other_data_preds[j]) / (np.linalg.norm(x_preds) * np.linalg.norm(other_data_preds[j]))
        elif corelation:
            x_preds = x_preds.reshape((-1,))
            distances[j] = pearsonr(x_preds, other_data_preds[j])[0]

    nns_idx = np.argsort(distances)
    nns_data = other_data[nns_idx]
    nns_lbs = other_labels[nns_idx]



    return nns_data, nns_lbs, nns_idx



def find_vnb_idx_new(index, test_data, test_labels, model):
    """select vulnerable x.寻找脆弱的测试集中的数据作为实验对象。
    Args:
        index: the index of train_data
        test_data: test data
        test_labels: test labels
        model:      
    Returns:
        vnb_idx: new idx sorted according to the vulnerability of data(more neighbors in same class, more vulnerable )


    """
    logging.info('Start select the vulnerable x')
    if os.path.exists(args.vnb_idx_path_new): # 如果已经保存过的话就直接打开使用
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

def find_x_i(model, x, y, x_train, y_train):
    logging.info('Start select x_i')
    if os.path.exists(args.x_i_information):
        x_i_idx = np.loadtxt(open(args.x_i_information, "r"), delimiter=",", skiprows=1)

        x_i_idx = x_i_idx[:, 0].astype(np.int32)
        logging.info(
            args.x_i_information + " already exist! Index of x_i have been restored from this file.")
        logging.info('The x_i index is: {}, length: '.format(x_i_idx[:20], len(x_i_idx)))

    else:
        logging.warn(
            args.x_i_information + " does not exist! Index of vulnerable x is generated for a long time ...")

        x_i, x_i_lbs, x_i_idx = get_nns(x, x_train, y_train, model)  #sort x_train wrt distances with x
        print('xi_idx:',x_i_idx[:20], 'length:', len(x_i_idx))



        x_i_idx_wo_own=[]
        for i in x_i_idx:
            if y_train[i] != y:
                x_i_idx_wo_own.append(i)
        print('xi_idx　without own class:',x_i_idx_wo_own[:20],'length:', len(x_i_idx_wo_own))
        print(y_train[x_i_idx_wo_own[:20]])
        x_i_idx = x_i_idx_wo_own
        x_i = x_train[x_i_idx_wo_own]

        x_i_from_target_train = 1
        if x_i_from_target_train:
            x_i_idx_wo_own_target = []
            for i in x_i_idx:
                if y_train[i] == args.target_class:
                    x_i_idx_wo_own_target.append(i)

            x_i_idx = x_i_idx_wo_own_target
            x_i = x_train[x_i_idx]
            print('xi_idx　without own&target class:', x_i_idx[:20], 'length:', len(x_i_idx))

        print(np.array(x_i_idx).reshape(len(x_i_idx), 1).shape)
        print(np.array(y_train[x_i_idx]).reshape(len(x_i_idx), 1).shape)
        print(np.array(model.get_labels(x_i)).reshape(len(x_i_idx), 1).shape)
        print(np.array(model.predict(x_i)).reshape(len(x_i_idx), 3).shape)

        matrix_x_i = np.hstack((np.array(x_i_idx).reshape(len(x_i_idx), 1),
                                np.array(y_train[x_i_idx]).reshape(len(x_i_idx), 1),
                                np.array(model.get_labels(x_i)).reshape(len(x_i_idx), 1),
                                np.array(model.predict(x_i)).reshape(len(x_i_idx), 3)
                                         ))


        with open(args.x_i_information, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['idx', 'real labels', 'pred labels', 'probs'])
            f_csv.writerows(matrix_x_i)
    return x_i_idx

def cross_entrophy(a, y): 
    return -np.sum(np.nan_to_num(y*np.log(a)+(1-y)*np.log(1-a)))


def fgsm(x, x_train, y_train, model, x_i_idx):
    x_i = x_train[x_i_idx]
    x_i_placeholder = K.placeholder(shape=( x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3]))

    y_i = y_train[x_i_idx]
    y_i = keras.utils.to_categorical(y_i, 10)
    y_i_placeholder = tf.placeholder(tf.int32, shape=(None, 10))

    y_preds_placeholder = tf.placeholder(tf.int32, shape=(None, 10))

    x_ = np.expand_dims(x, axis=0)
    x_ = np.repeat(x_, len(y_i), axis=0)
    epsilon = (np.max(x) - np.min(x)) * 0.01
    x_ = x_ + epsilon * np.random.randn(x_i.shape[0], x_i.shape[1], x_i.shape[2], x_i.shape[3])
    loss = keras.losses.categorical_crossentropy(y_i_placeholder = y_i, y_preds_placeholder = model.predict(x_))
    grads = K.gradients(loss, x_)

    fn = K.function([X], K.gradients(Y, [X]))

    grads[grads > 0] = 1
    cgd_grads_01 = 0
    if cgd_grads_01:
        grads[grads < 0] = 0
    else:
        grads[grads < 0] = -1

    epsilon_2 = (np.max(x_i) - np.min(x_i)) * 0.01

    x_i = x_i + epsilon_2 * grads

    return x_i

def main():


    x_train, y_train, x_test, y_test = my_load_dataset(args.dataset)
    model = select_model('resnet')



    first = 0
    model.set_model_path('./mymodels/10_3classes/' + str(i) + '.h5')
    if first:
        model.train()
    else:
        model.model = load_model(model.model_path)
        # model.model = load_model('./mymodels/cifar10_10lamda_x_num_3463.h5')
        
        logging.warn('model is restored from {}'.format(model.model_path))
        # logging.warn('model is restored from ./mymodels/cifar10_10lamda_x_num_3463.h5')

        # model.output_test_acc()
        # model.output_per_acc()
        # load the original model from model_path to avoid train again and save time

        
    if args.transfor_learning: #迁移学习
        model_new = model
    else:
        model_new = select_model('vgg')
    # model_new.output_test_acc()

    print('original model finished!!!')

    # if args.slt_stb_ts_x: #从测试集中选择稳定的实验数据
    #     logging.info('Selecting stable x by retraininretraingg 10 times using the same training data.')
    #     index = find_stable_idx(x_test, y_test, model)
    #     logging.info('First 20 / {} index of stable x: \n{}'.format(len(index), index[:20]))
    #     exit(0)
    # else:
    #     index = range(len(x_test))
    #     logging.info('Selecting x in all testing data, First 20 index: \n{}'.format(index[:20]))

    # decide which index

    # if args.slt_vnb_tr_x:  #从训练集中选择脆弱的实验数据
    #     index = find_vnb_idx_new(index, x_test, y_test, model)
    #     logging.info('Successfully selected vulnerable data, First 20 index: \n{}'.format((index[:20])))
    #     exit(0)

    succ=0

    print('first 10 labels:\n', y_test[:50])  # print first 10 test labels

    index = [5354, 6958, 8364, 255, 8918, 689] # 已经选好的几个测试数据的索引
    
    print(y_test[index])

    for idx in index:

        print('x idx:',idx)
        x = x_test[idx]


        # model.model = load_model('./mymodels/cifar10_3lamda_x_num_3848.h5')
        print('labels of x:', y_test[idx])
        print('preded label of x:', model.get_labels(x))



        save_first_10_figs_only=0
        if save_first_10_figs_only:
            save_fig(x.astype(np.int32), img_dir+str(idx)+'.png')
            continue

        if args.slt_x_i:
            nb_class = 3
            if nb_class == 3:
                train_idx = []
                classes = [0, 4, 7]
                for i in range(len(x_train)):
                    if y_train[i] in classes:
                        train_idx.append(i)
                print('train_idx.shape:', len(train_idx))
                x_train = x_train[train_idx]
                y_train = y_train[train_idx]
                print('x_train.shape:', x_train.shape)

            x_i_idx = find_x_i(model, x, y_test[idx], x_train, y_train)
        exit(0) # 如果你只需要寻找x_i的话，那么到这里就可以停止了。
        
        if args.use_fgsm:
            print('using fgsm')
            model_new.set_model_path(model_dir + str(args.dataset) + '_fgsm_idx_' + str(idx) + '.h5')

            x_i = fgsm(x, x_train, y_train, model, x_i_idx)
            x_train_new = x_train
            y_train_new = y_train

            for i in x_i_idx[:10]:
                save_fig(x_train[i], './imgs/fgsm/')


        if args.use_lamda_x0 != 0:
            new_path = model_dir + str(args.dataset) + '_' + str(args.use_lamda_x0) + 'lamda_x_num_' + str(idx) + '.h5'
            # if args.transfor_learning:
            #     new_path = new_path[:]
            model_new.set_model_path(new_path)
            # to save models with differnet names
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
        # model_new.model = load_model(model_new.model_path)

        transfor_learning = True

        if not transfor_learning:
            model_new.train(only_train=True)
        else:
            # transfer learning / freeze some layers
            start_layer, end_layer = 2, len(model_new.model.layers)

            # t=0
            for freeze_layer in range(start_layer,end_layer+1, 3):
                for layer in model_new.model.layers[:freeze_layer]:
                    # weights = layer.get_weights()
                    # t += 1
                    # print('freeze_layer',t, 'weights.shape:',len(weights))
                    # W = weights[0]
                    # b = weights[1]
                    # print('W.shape:',W.shape,'b.shape:',b.shape)

                    layer.trainable = False

                for i in range(10):
                    print('this is {}th epoch'.format(i))
                    model_new.epochs = 3
                    reset_weights = True

                    if i==0 and freeze_layer==start_layer:
                        model_new.train(only_train=True,
                                        transform=True,
                                        last_epoch=150 + i * model_new.epochs,
                                        freeze_layer=freeze_layer,
                                        reset_weights=reset_weights)
                        model_new.model.summary()
                    else:
                        if reset_weights:
                            last_epoch = i * model_new.epochs
                        else:
                            last_epoch = 150 + i * model_new.epochs

                        model_new.train(transform=True,  last_epoch= last_epoch)



                    print('labels of x:', y_test[idx])
                    print('preded label of x:', model_new.get_labels(x))
                    if model_new.get_labels(x)==args.target_class:
                        print('successful!!!')
                        with open('./log/succ.txt', 'a+') as f:
                            f.write('idx:'+str(idx)+', lamda:'+str(args.use_lamda_x0)+', successful at '+str(i)+'th epoch, ' )
                            f.write('freeze_layer:'+str(freeze_layer)+'\n')
                        break


        print('\n*********Start report result*********')

        if args.use_watermark or args.use_saliency:
            changed_data_labels_before = model.get_labels(changed_data)
            print('\nchanged_data_labels_before', changed_data_labels_before[:5])

            changed_data_labels_after = model_new.get_labels(changed_data)
            print('\nchanged_data_labels_after', changed_data_labels_after[:5])

        # output results
        label_x_after = model_new.get_labels(x)
        if label_x_after == args.target_class:
            print('successful!!!')
            succ += 1
        else:
            print('fail...')

        print('label_x_before:', y_test[idx], '\nlabel_x_after:', label_x_after)

        print('success times:', succ)

    #----------------------------------------
if __name__ == '__main__':
    main()







