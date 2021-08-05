"""
chapter7. 시계열 분석
    7.5 LSTM
"""

'''
5. LSTM 셀 구현
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 케라스에서 발생하는 경고 메세지 제거

import tensorflow as tf
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


# 값 초기화
tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

batch_size = 128
total_words = 10000  # 전체 단어 집합 크기. ex) 단어들이 0부터 20000까지 인코딩되었다면 단어 집합 크기는 20001
max_review_len = 80
embedding_len = 100  # 임베딩 후의 단어의 차원


# 데이터셋 준비
# Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)

# 모델의 입력으로 사용하려면 모든 샘플 길이를 동일하게 맞춰주어야 함
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 넘파이 배열을 Dataset으로 변화
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(10000).batch(batch_size, drop_remainder=True)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # TensorSliceDataset
test_data = test_data.batch(batch_size, drop_remainder=True)  # BatchDataset
print('x_train_shape : ', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test_shape : ', x_test.shape)

sample = next(iter(test_data))
print(sample[0].shape)

# -------------------------------- 이상 RNN_Cell, RNN_Layer과 동일 ----------------------------------------------------

# 네트워크 생성
class LSTM_Build(tf.keras.Model):
    def __init__(self,units):
        super(LSTM_Build, self).__init__()

        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]

        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.RNNCell0 = tf.keras.layers.LSTMCell(units, dropout=0.5)  # units : 메모리 셀의 개수
        self.RNNCell1 = tf.keras.layers.LSTMCell(units, dropout=0.5)
        self.outlayer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.RNNCell0(word, state0, training)
            out1, state1 = self.RNNCell1(word, state1, training)

        x = self.outlayer(out1)
        prob = tf.sigmoid(x)

        return prob


# 모델 훈련
import time
units = 64
epochs = 4
t0 = time.time()

model = LSTM_Build(units)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

model.fit(train_data, epochs=epochs, validation_data=test_data, validation_freq=2)


# 모델 평가
print("훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy*100))
print("테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss={:.4f}, accuracy : {:.4f}%".format(loss,accuracy*100))
t1 = time.time()
print("시간 : ", t1-t0)
