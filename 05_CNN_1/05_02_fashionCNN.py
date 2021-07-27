"""
chapter5. 합성곱 신경망 1
 5.2 합성곱 신경망 맛보기
  1. 심층 신경망 :  loss : 0.3597, accuracy : 0.8731
  >> 2. 합성곱 신경망 : loss :
"""

'''
fashion_mnist 데이터셋

데이터 형태가 너무 헷갈려~

x_train.shape - (60000, 28, 28)  (데이터 개수, height, width)
x_test.shape - (10000, 28, 28)

X_train_final - (60000, 28, 28, 1)  (데이터 개수, height, width, 
X_test_final - (10000, 28, 28, 1)
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# 데이터 전처리
X_train_final = x_train.reshape((-1,28,28,1)) / 255.  # reshape...
X_test_final = x_test.reshape((-1,28,28,1)) / 255.


# 합성곱 네트쿼으를 이용한 모델 생성
model_with_conv = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    # 32 = 합성곱 필터 개수 / (3,3) = 커널의 행과 열 / same = 입력 이미지와 출력 이미지 크기가 동일 / input_shape = (행, 열, 채널 개수)
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_with_conv.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_with_conv.fit(X_train_final, y_train, epochs=5)

model_with_conv.evaluate(X_test_final, y_test, verbose=2)