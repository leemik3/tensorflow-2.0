"""
chapter8. 성능 최적화
    8.3 하이퍼파라미터를 이용한 성능 최적화
"""

'''
2. 드롭아웃 (dropout) 을 이용한 성능 최적화
'''

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


# 데이터셋 내려받기
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True
)

padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=padded_shapes)


# 데이터 배열로 전환
train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()  # 배열로 변환


# 드롭아웃이 적용되지 않은 모델
encoder = info.features['text'].encoder
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# 모델 훈련
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = model.fit(train_batches, epochs=5, validation_data=test_batches, validation_steps=30)


# 훈련 결과 시각화
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_batches.shuffle(BUFFER_SIZE)

history_dict = history.history
