"""
chapter3. 머신러닝 핵심 알고리즘
  3.2 비지도 학습
    >> 1. 군집 <<
        목표 : 데이터 그룹화
        주요 알고리즘 : Kmeans
        예시 : 사용자의 관심사에 따라 그룹화하여 마케팅에 활용

    2. 차원 축소
        목표 : 데이터 간소화
        주요 알고리즘 : 주성분 분석 (PCA)
        예시 : 데이터 압축, 중요한 속성 도출

    - 데이터 간 유사도 측정 방법 : 유클리드 거리, 맨해튼 거리, 민코프스키 거리, 코사인 유사도
"""

'''
1. K-평균 군집화

    - 주어진 데이터에 대한 군집화 목적
    - 사전에 몇 개의 클러스터를 구성할지 알 수 있을 때 유용
    
    - 지양해야 하는 경우
        1) 데이터가 비선형일 경우
        2) 군집 크기가 다를 경우
        3) 밀집도와 거리가 다를 경우
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 상품에 대한 연 지출 데이터 호출
data = pd.read_csv('../../data/chap3/data/sales data.csv')
data.head()  # 명목형 데이터 - Channel, Region, 연속형 데이터 - 나머지


# 연속형 데이터와 명목형 데이터로 분류
categorical_features = ['Channel', 'Region']  # 명목형 데이터
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
data.head()


# 데이터 전처리 (스케일링) - 연속형 변수의 범위 통일
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)


# 적당한 k 값 추출
Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal K')
plt.show()
