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
import cv2
import os, glob, shutil
from sklearn.cluster import KMeans

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
input_dir = '../data/chap11/data/pets'
glob_dir = input_dir + '/*.jpg'

images = [cv2.resize(cv2.imread(file), (224,224)) for file in glob.glob(glob_dir)]
# images는 길이가 47인 리스트, 각 원소들은 (224,224,3) 크기의 ndarray
# resize 하지 않으면 각 원소들은 (375,500,3)이 됨
paths = [file for file in glob.glob(glob_dir)]
images = np.array(np.float32(images).reshape(len(images),-1)/255)  # shape = (47(이미지 개수), 나머지) ,,,,


# 특성 추출
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
predictions = model.predict(images.reshape(-1,224,224,3))
pred_images = predictions.reshape(images.shape[0],-1)


# 클러스터링 구성
k = 2
kmodel = KMeans(n_clusters=k, random_state=728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('../data/chap11/data/output')
for i in range(k):
    os.makedirs('../data/chap11/data/output'+str(i))
for i in range(len(paths)):
    shutil.copy2(paths[i], "../data/chap11/data/output"+str(kpredictions[i]))

