"""
chapter3. 머신러닝 핵심 알고리즘
  3.2 비지도 학습
    >> 1. 군집 <<  2. 차원 축소
"""

'''
2. 밀도 기반 군집 분석 (DBSCAN)

- 일정 밀도 이상을 가진 데이터를 기준으로 군집 형성

- 주어진 데이터에 대한 군집화

- K-평균 군집화와 다르게 클러스터의 숫자를 모를 때 유용
- 이상치가 많을 때 유용
- 노이즈(주어진 데이터셋과 무관 / 무작위성 데이터)에 영향을 받지 않음
- K-평균 군집화에 비해 연산량은 많음
'''