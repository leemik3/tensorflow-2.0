# tensorflow-2.0



### 머신러닝 학습 과정
학습 단계 : 데이터 → 특성 추출 → 머신러닝 알고리즘 적용 → 분류 / 예측 모델(모형)
예측 단계 : 데이터 → 특성 추출 → 분류 / 예측 모델(모형) → 결괏값

<br/>

### 머신러닝 학습 알고리즘
1. 지도 학습
    1. 분류 : KNN, SVM, 결정 트리, 로지스틱 회귀
    2. 회귀 : 선형 회귀
2. 비지도 학습
    1. 군집 : K-means clustering, 밀도 기반 군집 분석(DBSCAN)
    2. 차원 축소 : 주성분 분석 (PCA)
3. 강화 학습

    : 마르코프 결정 과정 (MDP)

---

### 딥러닝 학습 과정
1. 데이터 준비 
2. 모델(모형) 정의 
   
    :  은닉층 개수와 과적합 조절
3. 모델(모형) 컴파일 
   
    : 활성화 함수, 옵티마이저, 손실 함수 선택
    : 연속형 데이터셋의 경우 평균제곱오차(MSE)
    : 이진 분류의 경우 cross entropy
4. 모델(모형) 훈련
   
    : epoch, batch size
5. 모델(모형) 평가

<br/>

### 딥러닝 학습 알고리즘
1. 지도 학습
    1. 합성곱 신경망 (CNN)
       
        : 이미지 분류, 이미지 인식, 이미지 분할
    2. 순환 신경망 (RNN) 
       
        : 시계열 데이터 분류
        
        : LSTM - 기울기 소멸 문제 개선 위해 게이트 3개(망각, 입력, 출력) 추가 
2. 비지도 학습
    1. 워드 임베딩
       
        : Word2Vec, GloVe
    2. 군집
3. 전이 학습 : BERT, MobileNetV2
   1. 사전 학습 모델 (pretraining model)
        : ELMO, VGG, Inception, MobileNet
4. 강화 학습
   
    : 마르코프 결정 과정 (MDP)
   
---

### 자료 유형
- 수치형 자료
- 연속형 자료
- 이산형 자료
- 범주형 자료
- 순위형 자료 : 순서 의미 O
- 명목형 자료 : 순서 의미 X

---

### 머신러닝 회귀 모델의 평가 지표

- MAE: Mean Absolute Error
- MSE: Mean Square Error
- RMSE: Root Mean Square Error
- MAPE: Mean Absolute Percentage Error
- MPE: Mean Percentage Error

reference : https://velog.io/@tyhlife/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%9A%8C%EA%B7%80-%EB%AA%A8%EB%8D%B8%EC%9D%98-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C

---

### 결측값 처리 방법

0. 결측치 처리 가이드 라인
    - 10% 미만 : 삭제 or 대치
    - 10% ~ 50% : regression or model based imputation
    - 50% 이상 : 해당 컬럼 제거
    

1. 제거하기 (Deletion)
    1. 전체 행 삭제 (Listwise)
    2. 단일 값 삭제 (Pairwise)


2. 보간하기 / 대치 (Imputation)
    1. 최빈값
    2. 중앙값
    3. 평균
    4. 조건부 대치
    5. 회귀분석을 이용한 대치 
    

3. 예측 기법

reference : https://wooono.tistory.com/103

---

### 활성화 함수

0. linear
- 입력 뉴런과 가중치로 계산된 결괏값이 그대로 출력

1. 시그모이드 함수 (sigmoid)
- 선형 함수의 결과를 0~1 사이의 비선형 형태로 변형
- 주로 로지스틱 회귀와 같은 (이진) 분류 문제를 확률적으로 표현하는데 사용
- 기울기 소멸 문제 (vanishing gradient problem)


2. 하이퍼볼릭 탄젠트 함수 (tanh)
- -1~1 사이의 비선형 형태로 전황
- 시그모이드애서 결괏값의 평균이 양수로 편향된 문제를 해결 
- 시그모이드의 최적화 과정이 느려지는 문제 개선
- 기울기 소멸 문제 


3. 렐루 함수 (ReLU)
- 경사 하강법에 영향을 주지 않아 학습 속도가 빠르다.
- 기울기 소멸 문제가 발생하지 않는다
- 은닉층에서 주로 사용
- 음수 값을 입력받으면 항상 0을 출력하기 떄문에 학습 능력이 감소한다.


4. 리키 렐루 함수 (Leaky ReLU)
- 렐루 함수의 문제를 해결


5. 소프트맥스 함수
- 보통 출력 노드의 활성화 함수로 많이 사용한다.

---

### 손실 함수

1. 평균 제곱 오차 (MSE)
- 실제 값과 예측 값의 차이를 제곱하여 평균을 낸 것
- 작을 수록 예측력이 좋음
- 회귀에서 주로 사용됨
```angular2html
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
```

2. 크로스 엔트로피 오차(CEE)
- 분류 문제에서 원핫 인코딩을 했을 떄만 사용할 수 있음
- 분류 문제에서는 데이터의 출력을 0과 1로 구분하기 위해 시그모이드를 주로 사용하는데, 
시그모이드 함수에 포함된 자연 상수 e 때문에 평균제곱오차를 적용하면 매끄럽지 않은 그래프가 되므로
  크로스 엔트로피 손실 함수를 사용한다. (자연 로그를 모델의 출력값에 취함)

---

### 딥러닝의 문제점과 해결 방안

1. 과적합 문제 (over-fitting)
- 드롭아웃

2. 기울기 소멸 문제
- 시그모이드나 하이퍼볼릭 탄젠트 대신 ReLU 함수를 사용

3. 성능이 나빠지는 문제
- 확률적 경사 하강법, 미니 배치 경사 하강법으로 해결
    1. 배치 경사 하강법 : 전체 훈련 데이터셋에 대한 오류를 구한 뒤 업데이트
    2. 확률적 경사 하강법 : 임의로 선택한 데이터에 대해 기울기 계산, 속도가 빠름, 정확도가 낮을 수 있음
    3. 미니 배치 경사 하강법 : 전체 데이터셋을 미니 배치로 나누고 미니 배치 하나마다 기울기 계산, 업데이트, 확률적 경사 하강법보다 안정적
    

    * 옵티마이저
    : 확률적 경사 하강법의 파라미터 변경 폭이 불안정한 문제를 해결하기 위해 학습 속도와 운동량을 조정한다
    : 모멘텀, 네스테로프 모멘텀, 아다그라드, 알엠에스프롭, 아다델타, 아담

---

### 딥러닝 알고리즘

1. 심층 신경망 (DNN)


2. 합성곱 신경망 (CNN)
- LeNet-5, AlexNet, VGG, GoogLeNet, ResNet

3. 순환 신경망 (RNN)
- 시계열 데이터와 같이 시간 흐름에 따라 변화하는 데이터를 학습
- 순환 : 자기 자신을 참조한다는 뜻, 현재 결과가 이전 결과와 연관이 있을 때
- 기울기 소면 문제로 학습이 제대로 되지 않는 문제 - LSTM

4. 제한된 볼츠만 머신
- 가시층과 은닉층으로 구성
- 차원 감소, 분류, 선형 회귀 분석, 협업 필터링, feature learning, topic modeling에 사용

5. 심층 신뢰 신경망

---

# 합성곱 신경망

### Conv1D, Conv2D, Conv3D 데이터 형태 정리


### height, width, channel, length, depth 관계 정리
ex) Conv2D(~, input_shape=(<행>,<열>,<채널 개수>))

---

# 가중치 초기화 방법
```angular2html
Conv2D(~, kernel_initializer = ~)
```
### 1. 확률 분포 기반의 가중치 초기화
특정한 확률 분포에 기반하여 랜덤한 값을 추출하여 가중치를 초기화
- 균일 분포
- 정규 분포

### 2. 분산 조정 기반의 초기화
확률 분포를 기반으로 추출한 값으로 가중치를 초기화하되, 이 확률 분포의 분산을 가중치별로 동적으로 조절한다.   
분산을 조절할 때는 해당 가중치에 입력으로 들어오는 텐서의 차원 (fan in)과 결괏값으로 출력하는 텐서의 차원(fan out)이 사용된다. 
- LeCun 초기화 방식 : 입력 값의 크기가 커질 수록 초기화 값의 분산을 작게 만든다.
    - lecun_uniform
    - lecun_normal
- Xavier 초기화 방식 : fan in과 fan out을 모두 고려하여 확률 분포 계산
    - glorot_uniform
    - glorot_normal
- He 초기화 방식 : Xavier의 한계를 극복하려고 제안된 기법. fan out 보다 fan in 에 집중한 가중치
    - he_unifrom
    - he_normal
    
---

# 짚고 넘어가는 것들
어려웠던 것, 몰랐던 것, 헷갈렸던 것, etc

### 2021.07.29
1. ```super().__init__()``` 
[클래스, 상속, 오버라이딩의 개념](https://github.com/leemik3/python/wiki/%ED%81%B4%EB%9E%98%EC%8A%A4(class))

2. ```Conv2D(kernel_initializer='he_normal')```
[가중치 초기화 방법 개념](https://github.com/leemik3/tensorflow-2.0/#%EA%B0%80%EC%A4%91%EC%B9%98-%EC%B4%88%EA%B8%B0%ED%99%94-%EB%B0%A9%EB%B2%95)
   
3. ```Conv2D(padding='valid)```   
padding='valid' : 패딩 없음   
padding='same' : 입력과 출력의 크기가 같도록 패딩

4.   
- log_dir = '../../data/chap6/img/log6-2/'   
- log_dir = '../../data/chap6/img/log6-2'   
- log_dir = 'D:\git\tensorflow-2.0\data\chap6\img\log6-1\train   
마지막에 '/' 여부, 상대 경로와 절대 경로가 어떤 차이? 

### 2021.07.30
1. ```MaxPooling2D(data_format='channels_last)``` : 입력 형식을 설정하는 파라미터   
channels_last : 입력 데이터 형식이  (배치 크기, 높이, 너비, 채널개수)  
channels_first : 입력 데이터 형식이 (배치 크기, 채널개수, 높이, 너비)
   
2. strides=2 와 strides=(2,2)
: Conv2D에서 차이 없는 듯? 정확히 모름

3. ```flow_from_directory```   
: return (x,y)   
x : (batch_size, target_size, channels) 크기의 이미지  
y : labels
