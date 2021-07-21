'''
chapter3. 머신러닝 핵심 알고리즘
  3.1 지도 학습
    >> 1. 분류 <<  2. 회귀
'''

'''
2. SVM (Support Vector Machine)

- 결정 경계 / 기준선을 정의하는 모델
- margin 
    : 최적의 결정 경계는 데이터를 올바르게 분리하면서 마진 크기가 최대화
    : hard margin : 이상치를 허용하지 않음
    : soft margin : 어느 정도 이상치를 허용    
  
- 주어진 데이터에 대한 분류 목적
- 커널을 적절히 선택하면 정확도가 상당히 좋음
- 텍스트를 분류할 때도 자주 사용

'''

from sklearn import svm, metrics, datasets, model_selection
import tensorflow as tf
import os

os.environ['TF_CPP_WIN_LOG_LEVEL'] = '0'
# 해당 환경 변수로 로깅을 제어 - 0 : 모든 로그 표시 / 1 - INFO 로그 필터링 / 2 - WARNING 로그 필터링 / 3 - ERROR 로그 추가 필터링

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.6, random_state=42)

svm = svm.SVC(kernel='linear', C=1.0, gamma=0.5)
# kernel - 선형, 비선형 분류
# C - 클수록 하드 마진, 작을수록 소프트마진
# gamma : 결정 경계 유연성, 높으면 훈련 데이터에 의존하여 과적합 초래

svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
score = metrics.accuracy_score(y_test, predictions)
print("정확도 : {0:f}".format(score))