"""
chapter6. 합성곱 신경망 1
    6.1 이미지 분류를 위한 신경망

    V - LeNet-5
    - AlexNet
    - VGGNet
    - GoogLenNet
    - ResNet
"""

'''
1. LeNet-5

- 수표에 쓴 손글씨 숫자를 인식하는 딥러닝 구조, 현재 CNN의 초석
- 합성곱과 다운 샘플링/풀링을 반복적으로 거치면서 마지막 완전연결층에서 분류 수행
'''


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator



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


# LeNet-5 클래스 생성 : 입력 값은 이미지, 출력 값은 클래스의 확률 벡터
num_classes = 2


class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()  # Sequential 클래스의 __init__ 메서드 호출 - 기반 클래스가 초기화됨

        self.add(Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=input_shape, padding='same'))
        self.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), activation='relu', padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
        self.add(Dense(nb_classes, activation='softmax'))

        self.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])


# LeNet-5 모델 생성
model = LeNet((100,100,3), num_classes)
model.summary()


# 파라미터 초기화 및 데이터 호출
EPOCHS = 100
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = "../../data/chap6/data/catanddog/train/"
valid_dir = "../../data/chap6/data/catanddog/validation"


# 이미지 데이터 증가
train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

train_generator = train.flow_from_directory(
    train_dir,
    target_size=(image_height,image_width),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    seed=1,
    shuffle=True,
    class_mode='categorical'
)

valid = ImageDataGenerator(rescale=1.0/255.0)

valid_generator = valid.flow_from_directory(
    valid_dir,
    target_size=(image_height, image_width),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    seed=7,
    shuffle=True,
    class_mode='categorical'
)

train_num = train_generator.samples
valid_num = valid_generator.samples


# 텐서보드에서 모델 훈련 과정 살펴보기
log_dir = '../../data/chap6/img/log6-1/'  # 로그 파일이 기록될 위치
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_num // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_num // BATCH_SIZE,
    callbacks=[tensorboard_callback],
    verbose=1
)

'''
텐서보드 작동 - 프롬프트에 입력

[Trouble Shooting]

>>> tensorboard --logdir=../../data/chap6/img/log6-1/
- 오류 발생

>> tensorboard --logdir=D:\git\tensorflow-2.0\data\chap6\img\log6-1\
- 절대 경로 입력해서 해결
'''


