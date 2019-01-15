#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jingnan Jia
@contact: jiajingnan2222@gmail.com
@file: optimize_x_i.py.py
@time: 18-12-15 15 下午4:28
这个函数的目的是为了得到针对x这个个体的对抗样本,而不是针对某个类别。
'''

import keras
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import numpy as np
from keras.datasets import mnist, cifar10
from keras_poison import my_load_dataset
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
import tensorflow as tf
import os
from keras.layers.core import Reshape
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

    model_name = 'vgg'
    if model_name == 'resnet':
        model_path = './mymodels/cifar10_resnet-20.h5'
    elif model_name == 'vgg':
        model_path = './mymodels/0.h5'

    model = load_model(model_path)
    model.summary()


    # Turn the image into an array.
    # 根据载入的训练好的模型的配置，将图像统一尺寸
    x_train, y_train, x_test, y_test = my_load_dataset('cifar10')
    print(y_test[:20])
    # num_classes = 10
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    mean = 120.707
    std = 64.15
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    idx = 9
    x = np.expand_dims(x_test[idx], 0)
    print('x_preded_label:{}, real_label:{}'.format(model.predict(x), y_test[idx]))

    # x_i = np.random.randint(0, 255, x.shape)


    layer = 5

    # reshaped = Reshape((-1, 1))(drop5_1)
    layer_output = K.function([model.input], [model.layers[layer].output])

    # x_i_value = np.random.randint(0,255,x.shape)
    len_xi = 100
    x_is = np.ones(shape=(len_xi, 32, 32, 3))
    x_A = np.ones(shape=(3,3))
    for nb in range(len_xi+1):
        x_i = x_test[nb].reshape((-1, 32, 32, 3)).astype(np.float32)
        save_fig(x_i, './imgs/' + str(nb) + '_ori.png')
        if nb == 0:
            t = 5
        else:
            t = 1
        for i in range(t):
            layer_output_x = layer_output([x])[0]
            # print('layer_output_x.shape:\n',layer_output_x)

            # loss = K.sum(K.sum(K.sum(keras.losses.mean_squared_error(model.layers[layer].output[0], target_probs_variables), axis = 1), axis=1), axis=0, keepdims=True)[0]
            target_to_x = 0
            target_to_class = 1

            if target_to_x:
                target_probs_variables = K.variable(value=layer_output_x)
                loss = K.sum(keras.losses.mean_squared_error(model.layers[layer].output[0], target_probs_variables))
            elif target_to_class:
                target_probs_variables = K.variable(value=np.array([0,0,0,0,1,0,0,0,0,0], dtype='float32'))
                loss = K.sum(keras.losses.mean_squared_error(model.layers[-1].output[0], target_probs_variables))

            grad, = K.gradients(loss, model.input)
            calculate_grad_func = K.function([model.input], [loss, grad])
            loss_value, grad_value = calculate_grad_func([x_i])

            # print(loss_value.shape, grad_value)
            # grad_value = np.max(grad_value) - np.min(grad_value)
            if target_to_x:
                epsilon = 0.01
                x_i = x_i - grad_value * epsilon
            elif target_to_class:
                grad_value[grad_value<0] = -1
                grad_value[grad_value>0] = 1
                epsilon = 0.1
                x_i = x_i - grad_value * epsilon
            print('before ite:', y_test[nb])
            print('after ite:', np.argmax(model.predict(x_i)))
            save_fig(x_i, './imgs/'+str(nb)+'.png')
        if nb == 0:
            x_A = x_i
        else:
            x_is[i-1] = x_i

    print('x_is.shape:', x_is.shape)
    exit(0)


if "__main__" == __name__:
    main()
