#! /usr/bin/env python3
import argparse
import os

import numpy as np
import gym
from gym import spaces
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

import time
import cv2
import matplotlib.pyplot as plt

import signal
import sys

class DeepracerAgent():
    def __init__(self, ckpt_path, add_depth=False, pre_processing=False):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(device_count = {'GPU': 0}, gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.ckpt_path = ckpt_path
        self.pre_processing = pre_processing
        self.restore_model()
        self.graph = tf.get_default_graph()
        self.add_depth = add_depth
        self.input = self.graph.get_tensor_by_name('input:0')
        self.output_4 = self.graph.get_tensor_by_name('output_4:0')
        self.output_3 = self.graph.get_tensor_by_name('output_3:0')
        self.output_2 = self.graph.get_tensor_by_name('output_2:0')
        self.output_1 = self.graph.get_tensor_by_name('output_1:0')
        self.output_0 = self.graph.get_tensor_by_name('output_0:0')

    def restore_model(self):
        graph_def = tf.get_default_graph().as_graph_def()
        with gfile.FastGFile(self.ckpt_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        # tf_new_input = tf.placeholder(shape=(None, 120, 180, 3), dtype='float32')
        # tf.import_graph_def(graph_def, input_map={'input:0': tf_new_input}, name='')

        tf.import_graph_def(graph_def, name='')

    def act(self, obs, pre_processing=None):
        # img = tf.image.resize(obs['camera'], [480, 480])
        # print(img)
        # img = tf.reshape(img, [-1, 480, 480, 3])
        # print(img)
        #img = cv2.resize(obs['camera'], (480, 480))
        #img = cv2.resize(obs['camera'], (480, 320))
        img = cv2.resize(obs['camera'], (160, 120))
        img = img[:, :, :3] - [103.939, 116.779, 123.68] 
        img = [img]
        # em_maps = self.sess.run([self.output_0, self.output_1, self.output_2, \
                                 # self.output_3, self.output_4], feed_dict={
                                     # self.input: img})

        # 160, 120
        em_maps = self.sess.run([self.output_0, self.output_1, self.output_2, \
                                 self.output_3], feed_dict={
                                     self.input: img})
        for em in em_maps:
            print(em.shape)

        em_maps = np.array(em_maps)
        print(em_maps.shape)
        edge = np.mean(np.array(em_maps), axis=0)[0]

        #print(edge.shape)
        return edge


def sigint_handler(sig, frame):
    print('Got signal: ', sig)
    sys.exit(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='(string) path to the checkpoint to restore the model from.')
    args = parser.parse_args()

    agent = DeepracerAgent(args.checkpoint_path)

    img = cv2.imread("/home/hpc/nctu/hed/holy-edge/to_pb/000000.bmp")
    obs = {"camera": img}
    edge = agent.act(obs)
    cv2.imwrite("/home/hpc/nctu/hed/holy-edge/to_pb/result.jpg", np.uint8(255*edge))
