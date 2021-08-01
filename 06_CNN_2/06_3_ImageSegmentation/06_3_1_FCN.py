# 코드 없음

"""
chapter6. 합성곱 신경망 1
    6.3 이미지 분할을 위한 신경망

    V - 완전 합성곱 네트워크 (FCN)
    - 합성곱 & 역합성곱 네트워크 (convolutional & deconvolutional network)
    - U-Net
    - PSPNet
    - DeepLabv3/DeepLabv3+
"""

'''
1. 완전 합성곱 네트워크

완전연결층의 한계 : 고정된 크기의 입력만 받아들이며, 완전연결층을 거친 후에는 위치 정보가 사라진다. 
-> 따라서 완전연결층을 1X1 합성곱으로 대체함

위치 정보를 확인할 수 있음!!
'''
