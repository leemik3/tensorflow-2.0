"""
chapter6. 합성곱 신경망 1
    6.1 이미지 분류를 위한 신경망

    - LeNet-5
    V - AlexNet
    - VGGNet
    - GoogLenNet
    - ResNet
"""

'''
2. AlexNet

- GPU 2개를 기반으로 한 병렬 구조

'''

# 필요한 라이브러리 호출
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모델 생성

num_classes = 2
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super.__init__()

        self.add(Conv2D(96, kernel_size=(11,11), strides=4, padding='valid'), activation='relu',
                 input_shape=input_shape, kernel_initializer='he_normal')