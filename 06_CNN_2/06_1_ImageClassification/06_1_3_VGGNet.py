"""
chapter6. 합성곱 신경망 1
    6.1 이미지 분류를 위한 신경망

    - LeNet-5
    - AlexNet
    V - VGGNet
    - GoogLenNet
    - ResNet
"""

'''
3. VGGNet

- 합성곱층의 파라미터 개수를 줄이고 훈련 시간을 개선하려고 탄생.
- 네트워크 계층의 총 개수에 따라 여러 유형의 VGGNet (VGG16, VGG19)이 있다.
- VGG16 : 모든 합성곱 커널의 크기는 3X3, 최대 풀링 커널의 크기는 2X2, strides=2
'''

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2

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


# VGG19 네트워크 생성
class VGG19(Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu', input_shape=input_shape))
        self.add(Conv2D(64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.add(Conv2D(128, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(128, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=1))
        self.add(Conv2D(256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=1))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=2))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(1000, activation='softmax'))

        self.compile(optimizer=tf.keras.optimizers.Adam(0.003),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])


# VGG19 모델 출력
model = VGG19(input_shape=(224,224,3))
#model.summary()
# 파라미터가 1억 4000만 개라서 훈련하는 데 시간이 오래 걸린다.


# 사전 훈련된 VGG19 가중치 내려받기 및 클래스 정의
model.load_weights("../../data/chap6/data/vgg19_weights_tf_dim_ordering_tf_kernels.h5")
classes = {282 : 'cat',
           681 : 'notebook, notebook computer',
           970 : 'alp'}


# 이미지 호출 및 예측
image1 = cv2.imread("../../data/chap6/data/starrynight.jpeg")

