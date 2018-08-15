import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hello = tf.constant('Hello, TensorFlow!')

gpu_fraction = 0.1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print(sess.run(hello))


'''
CUDA_VISIBLE_DEVICES=1 python check_gpu.py
'''
