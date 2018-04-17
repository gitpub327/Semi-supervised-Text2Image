import tensorflow as tf
import tensorlayer as tl
import os
import random
import scipy
import scipy.misc
import numpy as np
import pickle as pkl
import re
import string

""" The functions here will be merged into TensorLayer after finishing this project.
"""

def load_data(rate):
    train_i,train_t,eval_i,eval_t,test_i,test_t = tl.files.load_mnist_dataset(
                                                        shape=(-1,28,28,1))

    train_i = np.asarray(train_i, dtype=np.float32)
    train_t = np.asarray(train_t, dtype=np.int32)
    eval_t = np.asarray(eval_t, dtype=np.float32)
    eval_t = np.asarray(eval_t, dtype=np.int32)
    test_i = np.asarray(test_i, dtype=np.float32)
    test_t = np.asarray(test_t, dtype=np.int32)

    
    indices = np.random.RandomState(seed=42).permutation(train_i.shape[0])
    train_i = train_i[indices]
    train_t = train_t[indices]
    indices2 = np.random.RandomState(seed=56).permutation(train_i.shape[0])
    wrong_i = train_i[indices2]
    wrong_t = train_t[indices2]

    indices3 = np.random.RandomState(seed=345).permutation(test_i.shape[0])
    test_i = test_i[indices3]
    test_t = test_t[indices3]

    indices5 = np.random.RandomState(seed=199).permutation(eval_i.shape[0])
    eval_i = eval_i[indices5]
    eval_t = eval_t[indices5]
    # np.random.shuffle(eval_i)
    # np.random.shuffle(eval_t)
    # np.random.shuffle(test_i)
    # np.random.shuffle(test_t)

    slice_num = int(train_i.shape[0] * rate)
    unlab_train_i = train_i[:slice_num]
    indices6 = np.random.RandomState(seed=457).permutation(unlab_train_i.shape[0])
    unlab_wrong_i = unlab_train_i[indices6]
    
    train_i = train_i[slice_num:-1]
    train_t = train_t[slice_num:-1]
    train_t = np.expand_dims(train_t, axis=1)
    test_t = np.expand_dims(test_t, axis=1)
    eval_t = np.expand_dims(eval_t, axis=1)
    wrong_t = np.expand_dims(wrong_t, axis=1)
    print("===Labeled sample: %d. Unlabeled sample: %d.===" % 
                            (train_i.shape[0],unlab_train_i.shape[0]))

    return unlab_wrong_i,wrong_t,wrong_i,train_i,train_t,unlab_train_i,test_i,test_t,eval_i,eval_t

def load_data2(rate):
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    train_i = mnistm['train']
    test_i = mnistm['test']
    eval_i = mnistm['valid']
    train_t = mnistm['raw']
    test_t = mnistm['test_raw']
    eval_t = mnistm['val_raw']
    print(train_i.shape[0])
    print(train_t.shape[0])
    print(test_i.shape[0])

    # train_i,train_t,eval_i,eval_t,test_i,test_t = tl.files.load_mnist_dataset(
    #                                                     shape=(-1,28,28,1))

    train_i = np.asarray(train_i, dtype=np.float32) /255.0
    train_t = np.asarray(train_t, dtype=np.int32).reshape(train_t.shape[0],28,28,1)
    eval_i = np.asarray(eval_i, dtype=np.float32)/255.0
    eval_t = np.asarray(eval_t, dtype=np.int32).reshape(eval_t.shape[0],28,28,1)
    test_i = np.asarray(test_i, dtype=np.float32)/255.0
    test_t = np.asarray(test_t, dtype=np.int32).reshape(test_t.shape[0],28,28,1)

    
    indices = np.random.RandomState(seed=42).permutation(train_i.shape[0])
    train_i = train_i[indices]
    train_t = train_t[indices]
    indices2 = np.random.RandomState(seed=56).permutation(train_i.shape[0])
    wrong_i = train_i[indices2]
    wrong_t = train_t[indices2]

    indices3 = np.random.RandomState(seed=345).permutation(test_i.shape[0])
    test_i = test_i[indices3]
    test_t = test_t[indices3]

    indices5 = np.random.RandomState(seed=199).permutation(eval_i.shape[0])
    eval_i = eval_i[indices5]
    eval_t = eval_t[indices5]
    # np.random.shuffle(eval_i)
    # np.random.shuffle(eval_t)
    # np.random.shuffle(test_i)
    # np.random.shuffle(test_t)

    slice_num = int(train_i.shape[0] * rate)
    unlab_train_i = train_i[:slice_num]
    indices6 = np.random.RandomState(seed=457).permutation(unlab_train_i.shape[0])
    unlab_wrong_i = unlab_train_i[indices6]
    
    train_i = train_i[slice_num:-1]
    train_t = train_t[slice_num:-1]


    # train_t = np.expand_dims(train_t, axis=1)
    # test_t = np.expand_dims(test_t, axis=1)
    # eval_t = np.expand_dims(eval_t, axis=1)
    # wrong_t = np.expand_dims(wrong_t, axis=1)
    print("===Labeled sample: %d. Unlabeled sample: %d.===" % 
                            (train_i.shape[0],unlab_train_i.shape[0]))

    return unlab_wrong_i,wrong_t,wrong_i,train_i,train_t,unlab_train_i,test_i,test_t,eval_i,eval_t

def load_ckpt(sess,saver,checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

def save_images(images, size, image_path):
    # images = (images+1.)/2.
    puzzle = merge(images, size)
    return scipy.misc.imsave(image_path, puzzle)

def merge(images, size):
    
    if images.shape[3] == 1:
        images = np.squeeze(images)
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image
        return img
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1],images.shape[3]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w,:] = image
        return img
