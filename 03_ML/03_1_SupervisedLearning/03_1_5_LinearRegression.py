"""
chapter3. 머신러닝 핵심 알고리즘
  3.1 지도 학습
    1. 분류  >> 2. 회귀 <<
"""

'''
4. 로지스틱 회귀와 선형 회귀

2) 선형 회귀

    - 종속 변수 : 연속형 변수
    - 모형 탐색 방법 : 최소 제곱법
    - 모형 검정 : F-테스트, t-테스트
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('../../data/chap3/data/weather.csv')


# 데이터 간 관계를 시각화로 표현
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()


# x 변수 Mintemp로 y변수 Maxtemp를 예측하도록, 선형 회귀 모델 생성
X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# 회귀 모델에 대한 예측
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted' : y_pred.flatten()})
# 미경아 flatten 찾아봐라


# 검증 데이터셋을 사용한 회귀선 표현
plt.scatter(X_test, y_test, color='grey')  # 실제 데이터
plt.plot(X_test, y_pred, color='red', linewidth=2)  # 예측 직선
plt.show()


# 선형 회귀 모델 평가
print('평균제곱법 :', metrics.mean_squared_error(y_test, y_pred)) # MSE
print('루트 평균제곱법 :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# RMSE - 모든 기온 백분율에 대한 평균값(22.41), 루트 평균제곱법 값(4.12)
# 10% 이상 (4/22 = 18%)이므로 정확도는 높지 않음
