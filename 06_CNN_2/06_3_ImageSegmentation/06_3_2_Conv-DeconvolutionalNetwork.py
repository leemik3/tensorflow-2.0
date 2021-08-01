# 코드 없음

"""
chapter6. 합성곱 신경망 1
    6.3 이미지 분할을 위한 신경망

    - 완전 합성곱 네트워크 (FCN)
    V - 합성곱 & 역합성곱 네트워크 (convolutional & deconvolutional network)
    - U-Net
    - PSPNet
    - DeepLabv3/DeepLabv3+
"""

'''
2. 합성곱 & 역합성곱 네트워크 (convolutional & deconvolutional network)

완전 합성곱 네트워크의 단점
    - 여러 단계의 합성곱층과 풀링층을 거치면서 해상도가 낮아진다.
    - 낮아진 해상도를 복원하기 위해 업 샘플링 방식을 사용하기 때문에 이미지의 세부 정보들을 잃어버린다.
    
합성곱으로 특성 맵 크기를 줄이고
역합성곱으로 특성 맵 크기를 증가시킴

역합성곱 (upsampling)
    - 최종 출력 결과를 원래의 입력 이미지와 같은 크기를 만들고 싶을 때 사용
    - 시멘틱 분할 (semantic segmentation) 등에 활용할 수 있음
    - 작동 방식 : 각각의 픽셀 주위에 제로 패딩을 추가 -> 패딩된 것에 합성곱 연산 수행
'''
