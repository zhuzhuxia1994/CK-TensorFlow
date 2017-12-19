import tensorflow as tf 
import numpy as np 
import os


#train_dir = '/home/hrz/projects/tensorflow/My-TensorFlow-tutorials/cats_vs_dogs/data/train/'

def get_files(file_dir):


	angry = []
	label_angry = []
	happy = []
	label_happy = []
	surprised = []
	label_surprised = []
	disgusted = []
	label_disgusted = []
	fearful = []
	label_fearful = []
	sadness = []
	label_sadness = []
	for sub_file_dir in os.listdir(file_dir):
		if sub_file_dir == 'angry':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				angry.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_angry.append(0)
		elif sub_file_dir == 'disgusted':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				disgusted.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_disgusted.append(1)
		elif sub_file_dir == 'fearful':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				fearful.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_fearful.append(2)
		elif sub_file_dir == 'happy':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				happy.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_happy.append(3)
		elif sub_file_dir == 'sadness':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				sadness.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_sadness.append(4)
		elif sub_file_dir == 'surprised':
			for name in os.listdir(file_dir+'/'+sub_file_dir):
				surprised.append(file_dir+'/'+sub_file_dir+'/'+name)
				label_surprised.append(5)
	print('Already!!',len(label_angry))


	image_list = np.hstack((angry,disgusted,fearful,happy,sadness,surprised))
	label_list = np.hstack((label_angry,label_disgusted,label_fearful,label_happy,label_sadness,label_surprised))
	temp = np.array([image_list,label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)

	image_list = list(temp[:, 0])
	label_list = list(temp[:, 1])
	label_list = [int(i) for i in label_list]
    
    
	return image_list, label_list


def get_batch(image,label,image_W,image_H,batch_size,capacity):
	'''
	Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

	image = tf.cast(image,tf.string)
	label = tf.cast(label,tf.int32)

	input_queue = tf.train.slice_input_producer([image,label])
	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_contents,channels=3)

	image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)

	image = tf.image.per_image_standardization(image)

	image_batch,label_batch = tf.train.batch([image,label],
    											batch_size = batch_size,
    											num_threads = 64,
    											capacity = capacity)

	label_batch = tf.reshape(label_batch,[batch_size])
	image_batch = tf.cast(image_batch,tf.float32)

	return image_batch,label_batch
