"""
chapter5. 합성곱 신경망 1
    5.3 전이 학습 (Transfer Learning)

    - 전이 학습 : 이미지넷처럼 아주 큰 데이터셋을 써서 훈련된 모델의 가중치를 가져와 우리가 해결하려는 과제에 맞게 보정해서 사용하는 것

    [전이 학습을 위한 방법]
    1. 특성 추출 기법
    >> 2. 미세 조정 기법 <<
"""

'''
2. 미세 조정 기법 (fine tuning)

- 사전 훈련된 모델과 합성곱층, 데이터 분류기의 가중치를 업데이트하여 훈련시키는 방식
- GPU 사용 권장

[전략]
1. 데이터셋이 크고 사전 훈련된 모델과 유사성이 적을 경우
    - 모델 전체를 재학습
    - 모델 전체를 재학습 하는 경우는 데이터셋이 작으면 과적합이 발생할 수 있어서?

2. 데이터셋이 크고 사전 훈련된 모델과 유사성이 클 경우
    - 합성곱층의 뒷부분과 데이터 분류기를 학습
    - 데이터셋이 크기 때문에 전체를 학습시키는 것보다는 강한 특징이 나타나는 합성곱층의 뒷부분과 데이터 분류기만 새로 학습

3. 데이터셋이 작고 사전 훈련된 모델과 유사성이 작을 경우
    - 합성곱층의 일부분과 데이터 분류기를 학습
    - 데이터가 적기 때문에 일부 계층에 미세 조정 기법을 적용해도 효과가 없을 수 있다.

4. 데이터셋이 작고 사전 훈련된 모델과 유사성이 클 경우
    - 데이터 분류기만 학습
    - 데이터가 적기 때문에 많은 계층에 미세 조정 기법을 사용하면 과적합이 발생할 수 있다.
'''