import os
import numpy as np
import tensorflow as tf
import input_data
import model
import sys

N_CLASSES = 6
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 16
CAPACITY = 100000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

s_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+/CK+YuanTu'
T_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+/CK+X_mid'
logs_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+'
s_train, s_train_label = input_data.get_files(s_train_dir)
#print(s_train)
s_train_batch, s_train_label_batch = input_data.get_batch(s_train,
                                                          s_train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY) 
#print(s_train_label_batch)
T_train, T_train_label = input_data.get_files(T_train_dir)
    
T_train_batch, T_train_label_batch = input_data.get_batch(T_train,
                                                          T_train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY) 
#print(len(s_train),len(s_train_label),len(T_train),len(T_train_label))
#print(s_train_batch,s_train_label_batch,T_train_batch,s_train_label_batch)
train_logits = model.inference(s_train_batch,T_train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, s_train_label_batch)        
train_op = model.trainning(train_loss, learning_rate)
#correct_prediction = tf.equal(tf.argmax(train_logits,1),tf.argmax(y_,1))
labels_max = tf.reduce_max(s_train_label_batch)

sess = tf.Session()
print(sess.run(labels_max))