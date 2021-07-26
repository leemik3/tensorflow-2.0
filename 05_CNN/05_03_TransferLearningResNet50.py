"""
chapter5. 합성곱 신경망 1
    5.3 전이 학습 (Transfer Learning)

    - 전이 학습 : 이미지넷처럼 아주 큰 데이터셋을 써서 훈련된 모델의 가중치를 가져와 우리가 해결하려는 과제에 맞게 보정해서 사용하는 것

    [전이 학습을 위한 방법]
    1. 특성 추출 기법
    2. 미세 조정 기법
"""

'''
1. 특성 추출 기법 (Feature Extractor)
- ImageNet 데이터셋으로 사전 훈련된 모델을 가져온 후 마지막에 완전연결층 부분만 새로 만든다.
- 학습할 때는 마지막 완전 연결층만 학습하고 나머지 계층은 학습되지 않도록 한다.

- 합성곱층 : 합성곱층과 풀링층으로 구성
- 데이터분류기 (안전 연결층) : 추출된 특성을 입력 받아 최종적으로 이미지에 대한 클래스를 분류하는 부분
--> 사전 훈련된 네트워크의 합성곱층 (가중치 고정)에 새로운 데이터를 통과시키고, 그 출력을 데이터 분류기에서 훈련 시킨다.

- 사용 가능한 이미지 분류 모델 : Xception, Inception V3, ResNet50, VGG16, VGG19, MobileNet
'''

# 예제
# ImageNet 데이터에 대해 사전학습된 ResNet50 모델을 사용
# ResNet50 : 계측 50개로 구성된 합성곱 신경망

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 사전 훈련된 ResNet50 모델 내려받기
model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# include_top : 네트워크 상단에 완전연결층을 포함할지 여부

# model.summary()

# ResNet50 네트워크에 밀집층 추가
model.trainable = False
model = Sequential([model,
                    Dense(2, activation='sigmoid')])
model.summary()


# 훈련에 사용될 환경 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 모델 훈련
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = '../data/chap5/data/catanddog/train'
valid_dir = '../data/chap5/data/catanddog/validation'

# ImageDataGenerator
train = ImageDataGenerator(
    rescale=1./255,  # rgb 계수를 0~1로 변환
    rotation_range=10,  # 0~10도 범위로 임의 회전
    width_shift_range=0.1,  # 그림 수평 랜덤 평행 이동
    height_shift_range=0.1,  # 그림 수직 랜덤 평행 이동
    shear_range=0.1,  # 임의로 전단 변형
    zoom_range=0.1)  # 임의 확대 축소 변형

# flow_from_directory: 폴더 구조를 그대로 가져와서 ImageDataGenerator 에 실제 데이터 채워줌
train_generator = train.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    seed=1,
    shuffle=True,
    class_mode='categorical')

valid = ImageDataGenerator(rescale=1.0/255.0)

valid_generator = valid.flow_from_directory(
    valid_dir,
    target_size=(image_height, image_width),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    seed=7,
    shuffle=True,
    class_mode='categorical')

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator,
    verbose=2)


# 모델의 정확도 시각화
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import font_manager

font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()

plt.rcParams['font.family'] = font_family

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, label='훈련 데이터셋')
plt.plot(epochs, val_accuracy, label='검증 데이터셋')
plt.legend()
plt.title('정확도')
plt.figure()

plt.plot(epochs, loss, label='훈련 데이터셋')
plt.plot(epochs, val_loss, label='검증 데이터셋')
plt.legend()
plt.title('오차')

# 훈련된 모델의 예측
class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))
prediction_values = model.predict_classes(validation)

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i, :], cmap=plt.cm.gray_r, interpolation='nearest')
    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17,  class_names[prediction_values[i]], color='red', fontsize=14)