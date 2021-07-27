"""
chapter5. 합성곱 신경망 1
    5.3 전이 학습 (Transfer Learning)

    - 전이 학습 : 이미지넷처럼 아주 큰 데이터셋을 써서 훈련된 모델의 가중치를 가져와 우리가 해결하려는 과제에 맞게 보정해서 사용하는 것

    [전이 학습을 위한 방법]
    1. 특성 추출 기법
    >> 2. 미세 조정 기법 <<
"""

'''
2. 미세 조정 기법 (fine tuning)

- 사전 훈련된 모델과 합성곱층, 데이터 분류기의 가중치를 업데이트하여 훈련시키는 방식
- GPU 사용 권장

[전략]
1. 데이터셋이 크고 사전 훈련된 모델과 유사성이 적을 경우
    - 모델 전체를 재학습
    - 모델 전체를 재학습 하는 경우는 데이터셋이 작으면 과적합이 발생할 수 있어서?

2. 데이터셋이 크고 사전 훈련된 모델과 유사성이 클 경우
    - 합성곱층의 뒷부분과 데이터 분류기를 학습
    - 데이터셋이 크기 때문에 전체를 학습시키는 것보다는 강한 특징이 나타나는 합성곱층의 뒷부분과 데이터 분류기만 새로 학습

3. 데이터셋이 작고 사전 훈련된 모델과 유사성이 작을 경우
    - 합성곱층의 일부분과 데이터 분류기를 학습
    - 데이터가 적기 때문에 일부 계층에 미세 조정 기법을 적용해도 효과가 없을 수 있다.

4. 데이터셋이 작고 사전 훈련된 모델과 유사성이 클 경우
    - 데이터 분류기만 학습
    - 데이터가 적기 때문에 많은 계층에 미세 조정 기법을 사용하면 과적합이 발생할 수 있다.
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
