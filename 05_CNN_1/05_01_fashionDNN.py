"""
chapter5. 합성곱 신경망 1
 5.2 합성곱 신경망 맛보기
  >> 1. 심층 신경망 :  loss : 0.3597, accuracy : 0.8731
  2. 합성곱 신경망
"""

'''
fashion_mnist 데이터셋
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# 분류에 사용될 클래스 정의
class_names = ['T-shirt','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(25):
    plt.subplot(5, 5, i+1)  # (nrows,ncols,index), 5,5 크기의 그래프를 그리겠다.
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()


# 심층 신경망을 이용한 모델 생성 및 훈련
x_train, x_test = x_train / 255.0, x_test / 255.0  # 이유 : 최대최소값이 1,0 이기 때문 / 정규화?

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 이미지를 1차원 데이터로 변경
    tf.keras.layers.Dense(128, activation='relu'),  # 단순한 심층 신경망에서는 이미지의 공간적 특성이 무시되는 단점이 있었음.
    # 하지만 합성곱 신경망에서 밀집층을 사용하게 되면 밀집층 직전의 입력과 그 후의 출력만 완전연결층으로 만들기 떄문에 이미지의 공간 정보를 유지할 수 있음.
    tf.keras.layers.Dropout(0.2),  # 과적합 방지, 20%의 노드를 무작위로 0으로 만든다.
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 옵티마이저 : 손실 함수를 사용하여 구한 값으로 기울기를 구하고 네트워크의 파라미터를 학습에 어떻게 반영할지 결정하는 방법
# 손실 함수 : 최적화 과정에서 사용, 다수의 클래스를 사용하므로 위 손실 함수 사용

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)