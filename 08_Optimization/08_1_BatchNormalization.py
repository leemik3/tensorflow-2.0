"""
chapter8. 성능 최적화
    8.3 하이퍼파라미터를 이용한 성능 최적화
"""

'''
1. 배치 정규화 (Batch Normalization) 를 이용한 성능 최적화
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

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

# 데이터프레임에 데이터셋 저장
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df = df.astype(float)
df['label'] = iris.target
df['label'] = df.label.replace(dict(enumerate(iris.target_names)))  # label 컬럼의 데이터는 target_names로 들어가게 됨


# 원-핫 인코딩 적용
label = pd.get_dummies(df['label'], prefix='label')  # 원핫 인코딩
df = pd.concat([df, label], axis=1)
df.drop(['label'], axis=1, inplace=True)


# 데이터셋 분류
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]  # dataframe
X = np.asarray(X)  # ndarray, shape : (150,4)
y = df[['label_setosa', 'label_versicolor', 'label_virginica']]
y = np.asarray(y)


# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)


##################################
# 배치 정규화가 적용되지 않은 모델 생성
##################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

model1 = Sequential([
    Dense(64, input_shape=(4,), activation='relu'),  # 입력층은 (4,0)의 형태를 가지고 유닛 64개로 구성
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax'),
])

model1.summary()


# 모델 훈련
model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model1.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_split=0.25,
    batch_size=40,
    verbose=2
)


# 훈련 결과 시각화
import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()  # y축이 2개 있는 플롯
loss_ax.plot(history1.history['loss'], 'y', label='train loss')
loss_ax.plot(history1.history['val_loss'], 'r', label='val loss')
acc_ax.plot(history1.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history1.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.legend(loc='upper right')
plt.show()
# 결과
# 훈련 정확도 100%에 가깝고, 훈련 손실은 0에 가까운 값 (good)
# 검증 손실값은 증가 (bad)


# 정확도와 손실 정보 표현
loss_and_metrics = model1.evaluate(X_test, y_test)
print('## 손실과 정확도 평가 ##')
print(loss_and_metrics)



# ######################
# 배치 정규화가 적용된 모델
# ######################
from tensorflow.keras.initializers import RandomNormal, Constant
model2 = Sequential([
    Dense(64, input_shape=(4,), activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(
        momentum=0.95,
        epsilon=0.005,
        beta_initializer=RandomNormal(mean=0.0, stddev=0.05),
        gamma_initializer=Constant(value=0.9)
    ),
    Dense(3, activation='softmax'),
])

model2.summary()


# 모델 훈련
model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    X_train,
    y_train,
    epochs=1000,
    validation_split=0.25,
    batch_size=40,
    verbose=2
)


# 훈련 결과 시각화
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history2.history['loss'], 'y', label='train loss')
loss_ax.plot(history2.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history2.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history2.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='lower right')
acc_ax.legend(loc='upper right')
plt.show()


# 모델 평가
loss_and_metrics = model2.evaluate(X_test, y_test)
print("손실과 정확도 평가")
print(loss_and_metrics)