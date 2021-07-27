"""
chapter5. 합성곱 신경망 1
    5.4. 설명 가능한 CNN (Explainable CNN)

- 딥러닝 처리 결과를 사람이 이해할 수 있는 방식으로 제시하는 기술
- CNN 처리 과정 시각화

1) 필터에 대한 시각화
2) 특성 맵에 대한 시각화 - V
"""

'''
5.4.1 특성 맵 시각화
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# GPU 작업 중복 문제 - 싱글 gpu 사용 시 해결 방안
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# 새로운 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(100,100,3), activation='relu', kernel_size=(5,5), filters=32),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5,5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5,5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(activation='relu', kernel_size=(5,5), filters=64),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()


# 특성 맵 정의
ins = model.inputs
outs = model.layers[0].output
feature_map = Model(inputs=ins, outputs=outs)
feature_map.summary()


# 이미지 호출
img = cv2.imread('../data/chap5/data/cat.jpg')
plt.imshow(img)


# 이미지 전처리 및 특성 맵 확인
img = cv2.resize(img, (100,100))
input_img = np.expand_dims(img, axis=0)
print(input_img.shape)

feature = feature_map.predict(input_img)  # 이미지를 모델에 적용
print(feature.shape)
fig = plt.figure(figsize=(50,50))
for i in range(16):
    ax = fig.add_subplot(8, 4, i+1)
    ax.imshow(feature[0,:,:,i])


# 이미지를 모델에 적용 - 2번째 계층
ins = model.inputs
outs = model.layers[2].output
feature_map = Model(inputs=ins, outputs=outs)

img = cv2.imread('../data/chap5/data/cat.jpg')
img = cv2.resize(img,(100,100))
input_img = np.expand_dims(img, axis=0)

feature = feature_map.predict(input_img)  # shape : (1,44,44,64)
fig = plt.figure(figsize=(50,50))
for i in range(48):  # 64로 해도됨
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(feature[0,:,:,i])


# 이미지를 모델에 적용 - 6번째 계층
ins = model.inputs
outs = model.layers[6].output
feature_map = Model(inputs=ins, outputs=outs)

img = cv2.imread('../data/chap5/data/cat.jpg')
img = cv2.resize(img, (100, 100))
input_img = np.expand_dims(img, axis=0)

feature = feature_map.predict(input_img)  # shape : (1,44,44,64)
fig = plt.figure(figsize=(50, 50))
for i in range(48):  # 64로 해도됨
    ax = fig.add_subplot(8, 8, i + 1)
    ax.imshow(feature[0, :, :, i])


# 결론 : 출력층에 가까워질 수록 원래 형태가 아닌 이미지 특징만 전달