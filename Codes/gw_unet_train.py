import tensorflow as tf
import os

from models import generator, discriminator, flownet, initialize_flownet
from loss_functions import intensity_loss, gradient_loss
from v_utils import DataLoader, load, save, psnr_error

import cv2
import numpy as np
from glob import glob

dataset_name = 'vessel'
train_folder = '/dmount/Vessel/drawing_anno/Patient/train/image'
train_gtfolder = '/dmount/Vessel/drawing_anno/Patient/train/label'
test_folder = '/dmount/Vessel/drawing_anno/Patient/vessel/30_1_in'
test_gtfolder = '/dmount/Vessel/drawing_anno/Patient/vessel/30_1_out'

train_dirs = glob(os.path.join(train_folder, '*.jpg'))
train_gtdirs = glob(os.path.join(train_gtfolder, '*.jpg'))
test_dirs = glob(os.path.join(test_folder, '*.jpg'))
test_gtdirs = glob(os.path.join(test_gtfolder, '*.jpg'))


batch_size = 8
epochs = 500
iterations = (len(train_dirs)//8 + 1) * epochs
height, width = 256, 256


l_num = 2
alpha_num = 1
lam_lp = 1.0
lam_gdl = 1.0

trial = 3


summary_dir = 'v_summary/trial_{}'.format(trial)
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

snapshot_dir = 'v_snapshot/trial_{}'.format(trial)
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)



lr_bounds = [7000]
lr = [0.0001, 1e-05]

train_inimg = np.zeros((len(train_dirs), height, width, 3), dtype=np.float32)
train_gtimg = np.zeros((len(train_dirs), height, width, 3), dtype=np.float32)
test_inimg = np.zeros((len(test_dirs), height, width, 3), dtype=np.float32)
test_gtimg = np.zeros((len(test_dirs), height, width, 3), dtype=np.float32)

for i in range(len(train_inimg)):
    img = cv2.imread(train_dirs[i])
    img = cv2.resize(img, (height, width))
    train_inimg[i] = (img / 127.5) - 1.0
    
    img = cv2.imread(train_gtdirs[i])
    img = cv2.resize(img, (height, width))
    train_gtimg[i] = (img / 127.5) - 1.0
    
for i in range(len(test_inimg)):
    img = cv2.imread(test_dirs[i])
    img = cv2.resize(img, (height, width))
    test_inimg[i] = (img / 127.5) - 1.0
    
    img = cv2.imread(test_gtdirs[i])
    img = cv2.resize(img, (height, width))
    test_gtimg[i] = (img / 127.5) - 1.0
    
# define dataset
with tf.name_scope('dataset'):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_inimg).repeat().batch(batch_size)
    train_gtdataset = tf.data.Dataset.from_tensor_slices(train_gtimg).repeat().batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_inimg).repeat().batch(batch_size)
    test_gtdataset = tf.data.Dataset.from_tensor_slices(test_gtimg).repeat().batch(batch_size)
    
    """train_dataset = tf.data.Dataset.from_tensor_slices(train_inimg).repeat(epochs)
    train_gtdataset = tf.data.Dataset.from_tensor_slices(train_gtimg).repeat(epochs)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_inimg).repeat(epochs)
    test_gtdataset = tf.data.Dataset.from_tensor_slices(test_gtimg).repeat(epochs)
    """
    
    """train_dataset = tf.data.Dataset.from_tensor_slices(train_inimg)
    train_dataset.repeat()
    train_dataset.batch(batch_size)
    
    train_gtdataset = tf.data.Dataset.from_tensor_slices(train_gtimg)
    train_gtdataset.repeat()
    train_gtdataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_inimg)
    test_dataset.repeat()
    test_dataset.batch(batch_size)
    
    test_gtdataset = tf.data.Dataset.from_tensor_slices(test_gtimg)
    test_gtdataset.repeat()
    test_gtdataset.batch(batch_size)"""
    
    
    train_it = train_dataset.make_one_shot_iterator()
    train_gtit = train_gtdataset.make_one_shot_iterator()

    test_it = test_dataset.make_one_shot_iterator()
    test_gtit = test_gtdataset.make_one_shot_iterator()


    train_inputs = train_it.get_next()
    train_gt = train_gtit.get_next()

    test_inputs = test_it.get_next()
    test_gt = test_gtit.get_next()
    

# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_outputs = generator(train_inputs, layers=4, output_channel=3)
    train_psnr_error = psnr_error(gen_frames=train_outputs, gt_frames=train_gt)

# define testing generator function
with tf.variable_scope('generator', reuse=True):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)


# define intensity loss
if lam_lp != 0:
    lp_loss = intensity_loss(gen_frames=train_outputs, gt_frames=train_gt, l_num=l_num)
else:
    lp_loss = tf.constant(0.0, dtype=tf.float32)


# define gdl loss
if lam_gdl != 0:
    gdl_loss = gradient_loss(gen_frames=train_outputs, gt_frames=train_gt, alpha=alpha_num)
else:
    gdl_loss = tf.constant(0.0, dtype=tf.float32)

    
with tf.name_scope('training'):
    g_loss = tf.add_n([lp_loss * lam_lp, gdl_loss * lam_gdl], name='g_loss')

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=lr_bounds, values=lr)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')


# add all to summaries
tf.summary.scalar(tensor=train_psnr_error, name='train_psnr_error')
tf.summary.scalar(tensor=test_psnr_error, name='test_psnr_error')
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=lp_loss, name='intensity_loss')
tf.summary.scalar(tensor=gdl_loss, name='gradient_loss')
tf.summary.image(tensor=train_outputs, name='train_outputs')
tf.summary.image(tensor=train_gt, name='train_gt')
tf.summary.image(tensor=test_outputs, name='test_outputs')
tf.summary.image(tensor=test_gt, name='test_gt')
summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')


    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None
    
    while _step < iterations:
        print('Training generator...')
        _, _g_lr, _step, _lp_loss, _gdl_loss, _g_loss, _train_psnr, _summaries = sess.run(
            [g_train_op, g_lrate, g_step, lp_loss, gdl_loss, g_loss, train_psnr_error, summary_op])

        if _step % 10 == 0:
            print('GeneratorModel : Step {}, lr = {:.6f}'.format(_step, _g_lr))
            print('                 Global      Loss : ', _g_loss)
            print('                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_lp_loss, lam_lp, _lp_loss * lam_lp))
            print('                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})'.format( _gdl_loss, lam_gdl, _gdl_loss * lam_gdl))
            print('                 PSNR  Error      : ', _train_psnr)
        
        if _step % 1000 == 0:
            summary_writer.add_summary(_summaries, global_step=_step)
            print('Save summaries...')

        if _step % 5000 == 0:
            save(saver, sess, snapshot_dir, _step)

            
    print('Finish successfully!')
    save(saver, sess, snapshot_dir, _step)
