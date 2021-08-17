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
import matplotlib.pyplot as plt
import numpy as np


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


# 모델 생성
class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11,11), strides=4, padding='valid', activation='relu',
                        input_shape=input_shape, kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', data_format='channels_last'))
        self.add(Conv2D(256, kernel_size=(5,5), strides=1, padding='same', activation='relu',
                        kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', data_format='channels_last' ))
        self.add(Conv2D(384, kernel_size=(3,3), strides=1, padding='same', activation='relu',
                        kernel_initializer='he_normal'))
        self.add(Conv2D(384, kernel_size=(3,3), strides=1, padding='same', activation='relu',
                        kernel_initializer='he_normal'))
        self.add(Conv2D(256, kernel_size=(3,3), strides=1, padding='same', activation='relu',
                        kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', data_format='channels_last'))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))
        self.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# 모델 생성
num_classes = 2
model = AlexNet((100,100,3), num_classes)
model.summary()


# 데이터 호출 및 데이터셋 전처리 (증가)
EPOCHS = 100
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = "../../data/chap6/data/catanddog/train/"
valid_dir = "../../data/chap6/data/catanddog/validation"

train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1)

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

train_num = train_generator.samples
valid_num = valid_generator.samples


# 텐서보드 설정 및 모델 훈련
log_dir = '../../data/chap6/img/log6-2/'
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

>>> tensorboard --logdir=../../data/chap6/img/log6-2/
- 오류 발생

>> tensorboard --logdir=D:\git\tensorflow-2.0\data\chap6\img\log6-2\
- 절대 경로 입력해서 해결
'''


# 분류에 대한 예측 : 총 32장인데 8장만 보여줌
class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))

prediction_values = model.predict_classes(validation)

fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')

    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)

#plt.show()