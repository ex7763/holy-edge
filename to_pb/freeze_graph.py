#! /usr/bin/env python3
import argparse
import os
from pathlib import Path, PurePath

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.tools import freeze_graph

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(args):
    ckpt_path = PurePath(args.checkpoint_path)
    meta_path = str(ckpt_path) + '.meta'
    #output_graph = ckpt_path.parent / (args.output_filename + '.pb')
    output_graph = (args.output_filename + '.pb')

    freeze_graph.freeze_graph(
        input_graph='',
        input_saver='',
        input_binary=True,
        input_checkpoint=str(ckpt_path),
        #output_node_names=['output_0', 'output_1', 'output_2', 'output_3', 'output_4'],
        output_node_names='output_0,output_1,output_2,output_3,output_4',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=str(output_graph),
        clear_devices=True,
        initializer_nodes='',
        input_meta_graph=meta_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', help='path to checkpoint file', type=str)
    parser.add_argument('-o', '--output_filename', help='output pb filename', type=str, default="model")
    args = parser.parse_args()
    main(args)
