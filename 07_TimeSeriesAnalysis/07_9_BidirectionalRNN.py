"""
chapter7. 시계열 분석
    7.8 양방향 RNN
"""

'''
9. 양방향 RNN 구현
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

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


# 모델 생성 및 훈련
n_unique_words = 10000
maxlen = 200
batch_size = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=4, validation_data=[x_test, y_test])


# LSTM 모델 구조 확인 ,, 양방향 LSTM
model.summary()


# 모델 평가
(loss, acc) = model.evaluate(x_train, y_train, batch_size=384, verbose=1)
print("Training accuracy", model.metrics_names, acc)
print("Training accuracy", model.metrics_names, loss)
(loss, acc) = model.evaluate(x_test, y_test, batch_size=384, verbose=1)
print("Training accuracy", model.metrics_names, acc)
print("Training accuracy", model.metrics_names, loss)
