'''
chapter3. 머신러닝 핵심 알고리즘
  3.1 지도 학습
    >> 1. 분류 <<  2. 회귀
'''

'''
3. Decision Tree (결정 트리)

- 데이터를 분류하거나 결괏값을 예측하는 분석 방법

- 주어진 데이터에 대한 분류 목적
- 이상치가 많은 값으로 구성된 데이터셋을 다룰 때 사용하면 좋음
- 결정 과정이 시각적으로 표현되기 떄문에 어떤 방식으로 의사 결정되는지 알고 싶을 떄 유용

- 엔트로피 : 불확실성 수치, 높을수록 불확실성 높고 순도 낮음
- 지니 계수 : 불순도 측정 지표, 높을수록 데이터가 분산됨
> 지니 계수 엔트로피보다 계산이 빨라서 결정 트리에서 자주 사용함 (로그를 계산할 필요가 없기 때문)
'''

# 라이브러리 호출 및 데이터 준비
import pandas as pd

pd.set_option('display.max_columns', 20)

df = pd.read_csv("../data/chap3/data/titanic/train.csv", index_col='PassengerId')


# 데이터 전처리
df = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Survived']]  # 필요한 컬럼만 사용
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # map 함수 - 모든 컬럼의 데이터 값이 숫자형이 되었음~
df = df.dropna()
X = df.drop('Survived', axis=1)
y = df['Survived']


# 훈련과 검증 데이터셋으로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# 결정 트리 모델 생성
from sklearn import tree
model = tree.DecisionTreeClassifier()


# 모델 훈련
model.fit(X_train, y_train)


# 모델 예측
y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))  # 모델 예측 정확도 : 79%


# 혼동 행렬을 이용한 성능 측정
from sklearn.metrics import confusion_matrix
pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)

'''
결과)
                   Predicted Not Survival  Predicted Survival
True Not Survival                     116                  12
True Survival                          33                  62

>> 정확한 예측 116+62 = 178 > 틀린 예측 33+12 = 45
>> 정밀도, 재현율, 정확도 알 수 있음
'''