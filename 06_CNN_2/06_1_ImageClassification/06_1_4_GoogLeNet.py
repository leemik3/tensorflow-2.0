"""
chapter6. 합성곱 신경망 1
    6.1 이미지 분류를 위한 신경망

    - LeNet-5
    - AlexNet
    - VGGNet
    V - GoogLenNet
    - ResNet
"""

'''
4. GoogLeNet (Inception)

- 주어진 하드웨어 자원을 최대한 효율적으로 이용하면서 학습 능력은 극대화
- 깊고 넓은 신경망을 위해 인셉션(Inception) 모듈 추가
- 특징을 효율적으로 추출하기 위해 1X1, 3X3, 5X5의 합성곱 연산을 각각 수행
- 희소 연결(sparse connectivity) : 관련성이 높은 노드끼리만 연결 - 연산량 감소, 과적합 해결
'''