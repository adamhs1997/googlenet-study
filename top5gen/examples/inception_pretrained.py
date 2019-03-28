#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_pretrained.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import top5gen.examples.loader as loader
from top5gen.src.nets.googlenet import GoogLeNet

PRETRINED_PATH = '/home/w266ajh/Documents/googlenet.npy'
DATA_PATH = '/home/w266ajh/Documents/top5gen/data'
IM_CHANNEL = 3


# Create a testing GoogLeNet model
test_model = GoogLeNet(
    n_channel=IM_CHANNEL, n_class=1000, pre_trained_path=PRETRINED_PATH)
test_model.create_test_model()

def get_logits():
    return test_model.layers['logits']
  
def get_labels():
    return test_model.label

def get_top_five(data_path, ftype):
    # Read ImageNet label into a dictionary
    label_dict = loader.load_label_dict()
    # Create a Dataflow object for test images
    image_data = loader.read_image(
        im_name=ftype, n_channel=IM_CHANNEL,
        data_dir=data_path, batch_size=1)



    # Init save
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_results = []
        while image_data.epochs_completed < 1:
            # read batch files
            batch_data = image_data.next_batch_dict()
            # get batch file names
            batch_file_name = image_data.get_batch_file_name()[0]
            # get prediction results
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: batch_data['image']})
            # display results
            current_results = []
            for re_prob, re_label, file_name in zip(pred[0], pred[1], batch_file_name):
                current_results.append(file_name)
                for i in range(5):
                    current_results.append('{}: probability: {:.02f}, label: {}'
                          .format(i+1, re_prob[i], label_dict[re_label[i]]))

                total_results.append(current_results)


    return total_results

if __name__ == "__main__":
    test_pre_trained()

