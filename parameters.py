#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:24:40 2018

@author: jiajingnan
"""


import argparse
parse = argparse.ArgumentParser()

parse.add_argument("--slt_vnb_tr_x", default=1,
                   help="use slt_vnb_tr_x")


parse.add_argument("--slt_stb_ts_x", default=1,
                   help="use slt_stb_ts_x")
parse.add_argument("--dataset", default='cifar10', 
                   help="dataset name", choices=['mnist','cifar10'])
parse.add_argument("--wart_power", default=0.2, 
                   help="how much does the wartmark add")
parse.add_argument("--changed_ratio", default=0.2, 
                   help="how much data in class is changed")
parse.add_argument("--target_class", default=4, 
                   help="which class does x0 want to become", choices=[0,1,2,3,4,5,6,7,8,9])
parse.add_argument("--num_classes", default=10, 
                   help="how many classes")
parse.add_argument("--give_up_ratio", default=0.5, 
                   help="how much area do you want to give up when extract salliency")

#choose one method
parse.add_argument("--use_lamda_x0", default=3, 
                   help="use lamda x0 directly to get perfect model with theta k, passed value is the lamda value")
parse.add_argument("--use_watermark", default=0, 
                   help="use watermark")
parse.add_argument("--use_saliency", default=0, 
                   help="use saliency as watermark")

#log file paths
log_dir = "./log/"
parse.add_argument("--path_label_change_x", default=log_dir+"label_change_log_x.txt",
                   help="a txt file to store the changed condition of x0")
parse.add_argument("--path_labels_changed_data_before", default=log_dir+"labels_changed_data_before.txt",
                   help="a txt file to store labels of changed_data")
parse.add_argument("--path_acc_tar_class", default=log_dir+"acc_tar_class.txt",
                   help="a txt file to store the accuracy per class")
parse.add_argument("--path_acc_per", default=log_dir+"acc_per.txt",
                   help="a txt file to store the accuracy average")
parse.add_argument("--log_file", default=log_dir+"logging_file.txt",
                   help="a txt file to store the accuracy average")
parse.add_argument("--vnb_idx_path_new", default=log_dir+'vnb_idx_new.csv',
                   help="vnb_idx_path_new")
parse.add_argument("--vnb_idx_path_new_p", default=log_dir+'vnb_idx_new_p.csv',
                   help="vnb_idx_path_new")


#model file paths
model_dir = "./mymodels/"

parse.add_argument("--path_mnist_model", default=model_dir+"mnist_2c1d.h5",
                   help="a h5 file to store the mnist_2c1d model")
parse.add_argument("--path_cifar10_2c2d_model", default=model_dir+"cifar10_2c2d.h5",
                   help="a h5 file to store the cifar10_2c2d model")
parse.add_argument("--path_cifar10_vgg_model", default=model_dir+"cifar10_vgg.h5",
                   help="a h5 file to store the cifar10_vgg model")
parse.add_argument("--path_cifar10_resnet_model", default=model_dir+"cifar10_resnet-52.h5",
                   help="a h5 file to store the cifar10_resnet model")

#perfect model file paths
parse.add_argument("--path_mnist_model_perf", default=model_dir+"mnist_2c1d_perf.h5",
                   help="a h5 file to store the mnist_2c1d model")
parse.add_argument("--path_cifar10_2c2d_model_perf", default=model_dir+"cifar10_2c2d_perf.h5",
                   help="a h5 file to store the cifar10_2c2d model")
parse.add_argument("--path_cifar10_vgg_model_perf", default=model_dir+"cifar10_vgg_perf.h5",
                   help="a h5 file to store the cifar10_vgg model")
parse.add_argument("--path_cifar10_resnet_model_perf", default=model_dir+"cifar10_resnet_perf.h5",
                   help="a h5 file to store the cifar10_resnet model")
#img file paths
img_dir = "./imgs/"
# =============================================================================
# parse.add_argument("--path_mnist_model", default=args.img_dir+"sal/",
#                    help="a png file to store the saliency imgs")
# parse.add_argument("--path_cifar10_2c2d_model", default=args.img_dir+"cifar10_2c2d.h5",
#                    help="a h5 file to store the cifar10_2c2d model")
# parse.add_argument("--path_cifar10_vgg_model", default=args.img_dir+"cifar10_vgg.h5",
#                    help="a h5 file to store the cifar10_vgg model")
# =============================================================================

#parse.add_argument("--gb", help="params means gb",action="store_const",const='value-to-store')

args = parse.parse_args()
