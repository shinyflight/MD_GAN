from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import yaml
import torch

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class MD_GAN(object):
    def __init__(self, sess):

        ############################################################################
        #  Hyper-parameters
        ############################################################################
        os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU']
        input_dim = config['INPUT_DIM']
        batch_size = config['BATCH_SIZE']
        hidden_dim = config['HIDDEN_DIM']
