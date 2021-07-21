'''
chapter3. 머신러닝 핵심 알고리즘
  3.1 지도 학습
    >> 1. 분류 <<  2. 회귀
'''

'''
1. K-최근접 이웃 (KNN; k-nearest neighbor)

- 새로운 입력(분류되지 않은 검증 데이터)을 받았을 때 기존 클러스터에서 모든 데이터와 인스턴스 기반 거리를 측정한 후 
  가장 많은 속성을 가진 클러스터에 할당하는 분류 알고리즘
  
- 주어진 데이터에 대한 분류 목적
- 직관적이고 사용하기 쉬워 초보자에게 좋음
- 훈련 데이터 충분한 환경에서 사용하면 좋음

'''

# 라이브러리 호출 및 데이터 준비
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics  # 모델 성능 평가

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']  # 데이터셋에 할당할 열 이름

dataset = pd.read_csv("chap3/data/iris.data", names=names)


# 훈련과 검증 데이터셋 분리
X = dataset.iloc[:, :-1].values  # 그냥 iloc - Dataframe 형태 / .values - numpy.ndarray 형태
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler  # Data Scaling / Feature Scaling : 모든 컬럼의 데이터 범위를 같게
s = StandardScaler()  # 특성 스케일링, 평균이 0 표준편차가 1이 되도록 변환
s.fit(X_train)  # 추가된 코드
X_train = s.transform(X_train)
s.fit(X_test)  # 추가된 코드
X_test = s.transform(X_test)

# 모델 생성 및 훈련
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

# 모델 정확도
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("정확도 : {}".format(accuracy_score(y_test, y_pred)))


# 최적의 k 찾기 : k값을 1부터 10까지 순환

k = 10
acc_array = np.zeros(k)

for k in np.arange(1, k+1, 1):  # start, stop, step
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("정확도", max_acc, ", 최적의 K는", k+1)
