"""
chapter 11. 클러스터링
    11.2 클러스터링 알고리즘 유형
"""

'''
1. K-means clustering (K 평균 군집화)
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# 데이터셋 준비
input_dir = '..'