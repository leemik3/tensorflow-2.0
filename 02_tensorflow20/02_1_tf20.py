# tensorflow 2.0 기초 문법
# 전체 동작하는 코드 아님
# 대략적인 작동 과정 및 흐름만 파악

'''
1. 데이터 준비

    데이터 호출 방법 : 판다스, 텐서플로

    이미지 : 전처리 후 배치 단위로 분할하여 처리
    텍스트 : 임베딩 후 서로 다른 길이의 시퀀스를 배치 단위로 분할하여 처리
'''

# 임의 데이터셋
import tensorflow as tf
import numpy as np

x = np.random.sample((100, 3))
dataset = tf.data.Dataset.from_tensor_slices(x)


# 텐서플로에서 제공하는 데이터셋
import tensorflow_datasets as tfds

df = tfds.load('mnist', split='train', shuffle_files=True)


# 케라스에서 제공하는 데이터셋
import tensorflow as tf

data_train, data_test = tf.keras.datasets.mnist.load_data()
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test


# 인터넷에서 데이터셋을 로컬 컴퓨터에 내려받아 사용
import tensorflow as tf
url = 'https://storage.googleapi.com/download.tensorflow.org/data/illiad/butler.txt'
text_path = tf.keras.utils.get_file('butler.txt', origin=url)


'''
2. 모델 정의

    Sequential API 
    : 초보자가 주로 사용
    
    Functional API 
    : 다차원 입출력을 갖는 신경망 구현, 전문가용
    : 입력 데이터의 크기를 input()의 파라미터로 사용하여 입력층을 정의해주어야 함
    
    Model Subclassing API 
    : 다차원 입출력을 갖는 신경망 구현, 전문가용
    : functional api와 본질적으로 차이가 없지만 가장 자유롭게 모델 구축 가능
'''

# Functional API
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(5,))  # 입력층 : 열(feature) 5개를 입력으로 받음
x = Dense(8, activation='relu')(inputs)  # 은닉층 1
x = Dense(4, activation='relu')(x)  # 은닉층 2
x = Dense(1, activation='softmax')(x)  # 출력층
model = Model(inputs, x)


# Model Subclassing API
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block_1 = Dense(32, activation='relu')
        self.block_2 = Dense(4, activation='sigmoid') # 4 : num_classes

    def call(self, inputs):
        x = self.block_1(input)
        return self.block_2


'''
3. 모델 컴파일
    
    [사전 정의 파라미터]
    1) 옵티마이저 : 모델의 업데이트 방법 결정
    2) 손실 함수 : 훈련하는 동안 오차 측정, y가 연속형 변수일 때는 MSE 주로 사용
    3) 지표 : 모델의 성능 측정
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# sparse_categorical_crossentropy는 다중 분류에서 사용되는 손실 함수, accuracy는 훈련에 대한 정확도, 1에 가까울수록 좋은 모델


'''
4. 모델 훈련
'''

model.fit(data_train, labels_train, epochs=10, batch_size=100, validation_data=(data_test, labels_test), verbose=2)
# verbose : 학습 진행 상황을 보여줄 것인지 지정, 1 - 학습 진행 상황 볼 수 있음


'''
5. 모델 평가
'''

model.evaluate(data_test, labels_test, batch_size=32)


'''
6. 훈련 과정 모니터링

    텐서보드 : 각종 파라미터 값이 어떻게 변화하는 지 시각화하여 볼 수 있음
'''

log_dir = 'logs/fit/'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# log_dit : 로그가 저장될 디렉터리 위치
# histogram_freq : 1 - 모든 에포크마다 히스토그램 게산 활성화, 0 - default, 히스토그램 계산 비활성화

model.fit(x=data_train, y=labels_train, epochs=5, validation_data=(data_test, labels_test), callbacks=[tensorboard_callback])

# >> tensorboard --logdir=./logs/fit/ : 텐서보드 실행
# >> http://localhost:6006 에서 조회 가능


'''
7. 모델 사용
'''

# model.predict(data_test)
