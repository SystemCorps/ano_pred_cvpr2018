import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(object):
    def __init__(self, infolder, gtfolder, resize_h=256, resize_w=256):
        self.indir = infolder
        self.gtdir = gtfolder
        self.resize_h = resize_h
        self.resize_w = resize_w
        
        self.inImgs = glob.glob(os.path.join(self.indir, '*.jpg'))
        self.inImgs.sort()
        self.gtImgs = glob.glob(os.path.join(self.gtdir, '*.jpg'))
        self.gtImgs.sort()
        
    
    def __call__(self, batch_size):
        inImgs = self.inImgs
        gtImgs = self.gtImgs
        
        
        re_h = self.resize_h
        re_w = self.resize_w
        
        def img_gen():
            for i in range(len(inImgs)):
                inImg = np_load_frame(inImgs[i], re_h, re_w)
                gtImg = np_load_frame(gtImgs[i], re_h, re_w)
                
                trImg = np.concatenate((inImg, gtImg), axis=2)
            
                yield trImg
                
        dataset = tf.data.Dataset.from_generator(generator=img_gen,
                                                output_types=tf.float32,
                                                output_shapes=[re_h, re_w, 6])
        
        data = dataset.batch(batch_size)
        
        return data

    
def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                      each frame in gen_frames.

    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = tf.shape(gen_frames)
    num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])
    gt_frames = (gt_frames + 1.0) / 2.0
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = tf.square(gt_frames - gen_frames)

    batch_errors = 10 * log10(1 / ((1 / num_pixels) * tf.reduce_sum(square_diff, [1, 2, 3])))
    return tf.reduce_mean(batch_errors)



def diff_mask(gen_frames, gt_frames, min_value=-1, max_value=1):
    # normalize to [0, 1]
    delta = max_value - min_value
    gen_frames = (gen_frames - min_value) / delta
    gt_frames = (gt_frames - min_value) / delta

    gen_gray_frames = tf.image.rgb_to_grayscale(gen_frames)
    gt_gray_frames = tf.image.rgb_to_grayscale(gt_frames)

    diff = tf.abs(gen_gray_frames - gt_gray_frames)
    return diff


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
