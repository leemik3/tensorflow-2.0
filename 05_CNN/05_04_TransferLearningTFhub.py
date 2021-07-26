"""
chapter5. 합성곱 신경망 1
    5.3 전이 학습 (Transfer Learning)

    - 전이 학습 : 이미지넷처럼 아주 큰 데이터셋을 써서 훈련된 모델의 가중치를 가져와 우리가 해결하려는 과제에 맞게 보정해서 사용하는 것

    [전이 학습을 위한 방법]
    >> 1. 특성 추출 기법 <<
    2. 미세 조정 기법
"""

'''
1-1) 텐서플로 허브
- 구글에서 공개한 API
- 모델에서 재사용 가능한 부분을 게시, 검색, 사용하기 위한 api 제공


import tensorflow as tf
import tensorflow_hub as hub

model = tf.keras.Sequential([
    hub.KerasLayer('https://tf')
])

'''