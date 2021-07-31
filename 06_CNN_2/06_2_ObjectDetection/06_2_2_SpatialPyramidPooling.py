# 코드 없음

"""
chapter6. 합성곱 신경망 1
    6.2 객체 인식을 위한 신경망

    - R-CNN
    V - 공간 피라미드 풀링
    - Fast R-CNN
    - Faster R-CNN
"""

'''
2. 공간 피라미드 풀링

- 과거 : 완전 연결층을 위해 입력 이미지를 고정해야 한다. (crop, warp)
- 개선 : 입력 이미지의 크기에 관계 없이 합성곱층을 통과시키고, 완전 연결층에 전달되기 전에 특성맵들을 동일한 크기로 조절해주는 풀링층 적용
'''
