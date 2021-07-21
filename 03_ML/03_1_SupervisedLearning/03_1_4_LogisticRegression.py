"""
chapter3. 머신러닝 핵심 알고리즘
  3.1 지도 학습
    >> 1. 분류 <<  2. 회귀
"""

'''
4. 로지스틱 회귀와 선형 회귀
- 독립 변수 (예측 변수), 종속 변수 (기준 변수)

1) 로지스틱 회귀
    - 독립 변수와 종속 변수가 선형 관계를 가질 때 사용하면 유용
    - 복잡한 연산 과정이 없기 때문에 컴퓨팅 성능이 낮은 환경이나 메모리 성능이 좋지 않을 때 좋음
    
    - 단순 선형 회귀 : x 값이 1개
    - 다중 선형 회귀 : x 값이 여러 개
'''

from sklearn.datasets import load_digits
digits = load_digits()

print('Image Data Shape', digits.data.shape)
print('Label Data Shape', digits.target.shape)


# digits 데이터셋의 시각화
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):  # enumerate (인덱스)와 zip (두 원소)
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training : %i\n' % label, fontsize=20)
#plt.show()


# 훈련과 검증 데이터셋 분리 및 로지스틱 회귀 모델 생성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)


# 일부 데이터를 사용한 모델 예측
logisticRegr.predict(x_test[0].reshape(1, -1))  #(1, something) 으로 맞춤
logisticRegr.predict(x_test[0:10])


# 전체 데이터를 사용한 모델 예측
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('score : ', score)


# 혼동 행렬 시각화
import numpy as np
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score : {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
