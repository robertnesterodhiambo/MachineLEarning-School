from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class MLP(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(MLP, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32   # Width of the input images
        self.input_height = 32  # Height of the input images
        self.image_channels = 3  # Number of channels in the images (RGB)
        self.num_classes = len(classes)
        self.hidden_layer_size = 128
        
        # Initialize your Layers here.
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)  # Output layer

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3);
                       during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when
                           you're doing Part 2 of the assignment and this
                           function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes);
                 during training, it would be (batch_size, num_classes)
        """
        # Flatten the input images
        x = self.flatten(inputs)
        
        # Pass through the first dense layer
        x = self.dense1(x)
        
        # Output logits through the second dense layer
        logits = self.dense2(x)
        
        return logits
