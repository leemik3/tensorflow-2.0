# 학습 없고 네트워크 정의만 있음

"""
chapter6. 합성곱 신경망 1
    6.1 이미지 분류를 위한 신경망

    - LeNet-5
    - AlexNet
    - VGGNet
    - GoogLenNet
    V - ResNet
"""

'''
5. ResNet

- Deep Residual Learning for Image Recognition 논문에서 제안

- 깊어진 신경망을 효과적으로 학습하기 위한 방법으로 residual 개념 고안
- 깊이가 깊다고 해서 무조건 성능이 좋아지는 것은 아님.
- 이를 위해 residual block 도입 
    : 기울기가 잘 전파될 수 있도록 (기울기 소실 문제 방지) 일종의 shortcut (skip connection)을 만들어줌


[기존 신경망]
x               y
↓               ↑
H(x) : 얻는 것 목적



[ResNet]
x                        y
↓                        ↑
H(x) = F(x) + x : 최소화 목적


H(x) = F(x) + x
F(x) = H(x) - x = 출력 - 입력
F(x) = 레지듀얼 함수

F(x)를 0으로 보내는 것이 목적
H(x) = x로 보내는 것이 목적
H(x) - x 를 최소로
H(x) - x = 레지듀얼 


1. H(x) = x 가 되도록 학습
2. F(x)가 0이 되도록 학습
3. F(x) + x = H(x) = x 가 되도록 학습 -> F(x) + x의 미분 값은 F'(x) + 1로 최소 1 이상의 값이 됨
4. 모든 계층에서 기울기가 F'(x) + 1이므로 기울기 소멸 문제가 해결

┌ 아이덴티티 블록
└ 합성곱 블록
'''

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2


# 아이덴티티 블록
def res_identity(x, filters):
    x_skip = x
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2D(f1, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x = Add()([x, x_skip])  # 이게 왜 숏컷?
    x = Activation(activations.relu)(x)
    return x


# 합성곱 블록
def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2D(f1, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x_skip = Conv2D(f2, kernel_size=(1,1), strides=(s,s), padding = 'valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)
    return x


'''
Q. 그래서 x엔 뭐가 들어가나?
'''