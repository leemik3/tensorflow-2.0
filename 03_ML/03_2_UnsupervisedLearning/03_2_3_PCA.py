"""
chapter3. 머신러닝 핵심 알고리즘
  3.2 비지도 학습
    1. 군집  >> 2. 차원 축소 <<
"""

'''
3. 주성분 분석 (PCA)

- 주어진 데이터의 간소화 목적

- 데이터의 feature가 너무 많을 경우에는 하나의 plot에 시각화하기 어려움.
- 중요하지 않은 변수로 인해 처리할 데이터 양이 많아지고 성능이 나빠지는 문제를 위해서 대표 특성만 추출

1. 데이터들의 분포 특성을 잘 설명하는 벡터를 두 개 선택
2. 벡터 두 개를 위한 적정한 가중치를 찾을 때까지 학습 진행

pca = decomposition.PCA(n_components=1)
pca_x = pca.fit_transform(x_std)
result = pd.DataFrame(pca_x, columns=['dog'])
result['y-axis'] = 0.0
result['label'] = Y
sns.lmplot('dog', 'y-axis', data=result, fit_reg=False, scatter_kws={'s':50}, hue='label')

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os

os.environ['TF_CPP_WIN_LOG_LEVEL'] = '2'

X = pd.read_csv('../../data/chap3/data/credit card.csv')
X = X.drop('CUST_ID', axis=1)
X.fillna(method='ffill', inplace=True) # 결측값을 앞의 값으로 채움 미경공부
print(X.head())


# 데이터 전처리 및 데이터를 2차원으로 차원 축소
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 평균이 0 표준편차가 1

X_normalized = normalize(X_scaled)  # 가우스 분포
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)  # 2차원으로 차원 축소 선언
X_principal = pca.fit_transform(X_normalized)  # 차원 축소 적용
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())


# DBSCAN 모델 생성 및 결과의 시각화
db_fault = DBSCAN(eps=0.0375, min_samples=3).fit(X_principal)  # 모델 생성 및 훈련
labels = db_fault.labels_  #

colours = {}
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]  # labels : 8950개

r = plt.scatter(X_principal['P1'], X_principal['P2'], color='y')
g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g')
b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b')
k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k')

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)

plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))
plt.show()


# 모델 튜닝
db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours[1] = 'g'
colours[2] = 'b'
colours[3] = 'c'
colours[4] = 'y'
colours[5] = 'm'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[0])
g = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[1])
b = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[2])
c = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[3])
y = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[4])
m = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[5])
k = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.legend((r, g, b, c, y, m, k), ('label 0', 'label 1', 'label 3', 'label 4', 'label 5', 'label -1'), scatterpoints=1,
           loc='upper left', ncol=3, fontsize=8)
plt.show()


# min