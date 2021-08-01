"""
chapter6. 합성곱 신경망 1
    6.3 이미지 분할을 위한 신경망

    - 완전 합성곱 네트워크 (FCN)
    - 합성곱 & 역합성곱 네트워크 (convolutional & deconvolutional network)
    V - U-Net
    - PSPNet
    - DeepLabv3/DeepLabv3+
"""

'''
3. U-Net

바이오 메디컬 이미지 분할을 위한 합성곱 신경망

특징
- 속도가 빠르다 : 이미 검증이 끝난 패치는 건너뜀
- 트레이드오프(패치 크기가 커짐 = 컨텍스트 인식이 좋아짐 = 지역화에 한계가 있음)에 빠지지 않는다

구조
- FCN 기반
┌ 수축 경로 : 컨텍스트 포착, 
└ 확장 경로 : 특성 맵 업 샘플링, 수축 경로에서 포착한 컨택스트와 결합하여 정확한 지역화 수행
- 3 X 3 합성곱이 주를 이룸
'''

import tensorflow as tf
from tensorflow.keras import layers


# 수축 경로 구현 : 합성곱층 2개 + 2X2 maxpooling 1개로 구성

inputs = layers.Input(shape=(572,572,1))  # channel 1개 - grayscale 이미지

c0 = layers.Conv2D(64, activation='relu', kernel_size=3)(inputs)
c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(c0)
c2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(c1)

c3 = layers.Conv2D(128, activation='relu', kernel_size=3)(c2)
c4 = layers.Conv2D(128, activation='relu', kernel_size=3)(c3)
c5 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(c4)

c6 = layers.Conv2D(256, activation='relu', kernel_size=3)(c5)
c7 = layers.Conv2D(256, activation='relu', kernel_size=3)(c6)
c8 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(c7)

c9 = layers.Conv2D(512, activation='relu', kernel_size=3)(c8)
c10 = layers.Conv2D(512, activation='relu', kernel_size=3)(c9)
c11 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(c10)

c12 = layers.Conv2D(1024, activation='relu', kernel_size=3)(c11)
c13 = layers.Conv2D(1024, activation='relu', kernel_size=3)(c12)


# 확장 경로 구현

t01 = layers.Conv2DTranspose(512, kernel_size=2, strides=(2,2), activation='relu')(c13)
crop01 = layers.Cropping2D(cropping=(4,4))(c10)
concat01 = layers.concatenate([t01, crop01], axis=-1)

c14 = layers.Conv2D(512, activation='relu', kernel_size=3)(concat01)
c15 = layers.Conv2D(512, activation='relu', kernel_size=3)(c14)

t02 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2,2), activation='relu')(c15)
crop02 = layers.Cropping2D(cropping=(16,16))(c7)
concat02 = layers.concatenate([t02, crop02], axis=-1)

c16 = layers.Conv2D(256, activation='relu', kernel_size=3)(concat02)
c17 = layers.Conv2D(256, activation='relu', kernel_size=3)(c16)

t03 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2,2), activation='relu')(c17)
crop03 = layers.Cropping2D(cropping=(40,40))(c4)
concat03 = layers.concatenate([t03, crop03], axis=-1)

c18 = layers.Conv2D(128, activation='relu', kernel_size=3)(concat03)
c19 = layers.Conv2D(128, activation='relu', kernel_size=3)(c18)

t04 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2,2), activation='relu')(c19)
crop04 = layers.Cropping2D(cropping=(88,88))(c1)
concat04 = layers.concatenate([t04, crop04], axis=-1)

c20 = layers.Conv2D(64, activation='relu', kernel_size=3)(concat04)
c21 = layers.Conv2D(64, activation='relu', kernel_size=3)(c20)

outputs = layers.Conv2D(2, kernel_size=1)(c21)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='u-netmodel')