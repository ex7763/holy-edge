import os
import sys
import argparse
import yaml
import urllib.parse
import urllib
from io import StringIO
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from hed.models.vgg16 import Vgg16
from hed.utils.io import IO


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class HEDReshaper():

    def __init__(self, config_file):

        self.io = IO()
        self.init = True

        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def setup(self, session):

        try:

            self.model = Vgg16(self.cfgs, run='reshaping')

            meta_model_file = os.path.join(self.cfgs['save_dir'], 'models/hed-model-{}'.format(self.cfgs['test_snapshot']))

            saver = tf.train.Saver()
            saver.restore(session, meta_model_file)


            self.io.print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))

        except Exception as err:

            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self, session):

        if not self.init:
            return

        self.model.setup_reshaping(session)

        idx = 0
        saver = tf.train.Saver()
        saver.save(session, os.path.join(self.cfgs['save_dir'], 'reshape_models/hed-model'), global_step=idx)


