import os
import sys
import time
import wget
import numpy as np
from PIL import Image, ImageEnhance
from hed.utils.io import IO
import random


def random_flip_left_right(img, prob=0.3):
    p = random.uniform(0.0, 1.0)
    if p < prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_contrast(img, prob=0.3, lower = 0.2, upper = 1.8):
    p = random.uniform(0.0, 1.0)
    if p < prob:
        factor = random.uniform(lower, upper)
        img = ImageEnhance.Sharpness(img)
        img = img.enhance(factor)
    return img

def random_brightness(img, prob=0.3, lower = 0.6, upper = 1.4):
    p = random.uniform(0.0, 1.0)
    if p < prob:
        factor = random.uniform(lower, upper)
        img = ImageEnhance.Brightness(img)
        img = img.enhance(factor)
    return img

def random_color(img, prob=0.3, lower = 0.6, upper = 1.5):
    p = random.uniform(0.0, 1.0)
    if p < prob:
        factor = random.uniform(lower, upper)
        img = ImageEnhance.Color(img)
        img = img.enhance(factor)
    return img

class DataParser():

    def __init__(self, cfgs):

        self.io = IO()
        self.cfgs = cfgs
        self.train_file = os.path.join(cfgs['download_path'], cfgs['training']['list'])
        self.train_data_dir = os.path.join(cfgs['download_path'], cfgs['training']['dir'])
        self.training_pairs = self.io.read_file_list(self.train_file)

        self.samples = self.io.split_pair_names(self.training_pairs, self.train_data_dir)
        self.io.print_info('Training data set-up from {}'.format(os.path.join(self.train_file)))
        self.n_samples = len(self.training_pairs)

        self.all_ids = list(range(self.n_samples))
        np.random.shuffle(self.all_ids)

        self.training_ids = self.all_ids[:int(self.cfgs['train_split'] * len(self.training_pairs))]
        self.validation_ids = self.all_ids[int(self.cfgs['train_split'] * len(self.training_pairs)):]

        self.io.print_info('Training samples {}'.format(len(self.training_ids)))
        self.io.print_info('Validation samples {}'.format(len(self.validation_ids)))

    def get_training_batch(self):

        batch_ids = np.random.choice(self.training_ids, self.cfgs['batch_size_train'])

        return self.get_batch(batch_ids)

    def get_validation_batch(self):

        batch_ids = np.random.choice(self.validation_ids, self.cfgs['batch_size_val'])

        return self.get_batch(batch_ids)

    def get_batch(self, batch):

        tstart = time.time()

        filenames = []
        images = []
        edgemaps = []

        for idx, b in enumerate(batch):

            im = Image.open(self.samples[b][0])
            em = Image.open(self.samples[b][1])

            im = im.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))
            em = em.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))

            # Image Augmentation
            if np.random.random() < 0.3:
                im = random_flip_left_right(im, prob=1.)
                em = random_flip_left_right(im, prob=1.)
            im = random_contrast(im, prob=0.3, lower=0.2, upper=1.8)
            im = random_brightness(im, prob=0.3, lower=0.6, upper=1.4)
            im = random_color(im, prob=0.3, lower=0.6, upper=1.5)

            im = np.array(im, dtype=np.float32)
            im = im[:, :, self.cfgs['channel_swap']]
            im -= self.cfgs['mean_pixel_value']

            em = np.array(em.convert('L'), dtype=np.float32)

            # Labels needs to be 1 or 0 (edge pixel or not)
            # or can use regression targets as done by the author
            # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213
            if self.cfgs['target_regression']:
                bin_em = em / 255.0
            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1

            # Some edge maps have 3 channels some dont
            bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
            # To fit [batch_size, H, W, 1] output of the network
            bin_em = np.expand_dims(bin_em, 2)

            images.append(im)
            edgemaps.append(bin_em)
            filenames.append(self.samples[b])

        return images, edgemaps, filenames
