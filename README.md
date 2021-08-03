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

filter 개수 ≠ filter channel 수

---

### 기울기 소멸 / 소실 문제 (Vanishing Gradient Problem)


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
# 시계열 분석
- 특정 대상의 시간에 따라 변하는 데이터를 사용하여 추이를 분석하는 것. 
- 목적 : 추세를 파악하거나 향후 전망 등을 예측하기 위함
- EX) 주가/환율 변동, 기온/습도 변화
---
### 시계열 형태
- **불규칙적 시계열** : 트렌드 혹은 분산이 변화
    - 불규칙 변동 (irregular variation)
        - 규칙성이 없어 예측 불가능하고 우연적으로 발생하는 변동
        - ex) 전쟁, 홍수, 화재, 지진, 파업   
<br/>
- **규칙적 시계열** : 트렌드와 분산이 불변
    - 추세 변동 (trend variation)
        - 시계열 자료가 갖는 장기적인 변화 추세   
          (*추세 : 장기간에 걸쳐 지속적으로 증가/감소하거나 일정한 사태를 유지하려는 성향)
        - ex) 국내총생산(GDP), 인구증가율
    - 순환 변동 (cyclical variation)
        - 대체로 2~3년 정도의 일정한 기간을 주기로 순환적으로 나타나는 변동
        - 1년 이내 주기로 곡선을 그리며 추세 변동에 따라 변동하는 것
        - ex) 경기 변동
    - 계절 변동 (seasonal variation)
        - 보통 계절적 영향과 사회적 관습에 따라 1년 주기로 발생하는 것

시계열 데이터를 잘 분석한다 ?   
= 불규칙적 시계열 데이터에 특정한 기법이나 모델을 적용하여 규칙적 패턴을 찾거나 예측하는 것   

불규칙적 시계열 데이터에 규칙성을 부여하는 방법
- AR, MA, ARMA, ARIMA 모델 적용,
- 딥러닝을 이용하여 스스로 시계열 데이터의 연속성을 찾아내도록 하는 것

---
### 
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

4. 경로 - 마지막에 '/' 여부, 상대 경로와 절대 경로가 어떤 차이?
- log_dir = '../../data/chap6/img/log6-2/'   
- log_dir = '../../data/chap6/img/log6-2'   
- log_dir = 'D:\git\tensorflow-2.0\data\chap6\img\log6-1\train   
 

### 2021.07.30
1. ```MaxPooling2D(data_format='channels_last)``` : 입력 형식을 설정하는 파라미터   
channels_last : 입력 데이터 형식이  (배치 크기, 높이, 너비, 채널개수)  
channels_first : 입력 데이터 형식이 (배치 크기, 채널개수, 높이, 너비)
   
2. strides=2 와 strides=(2,2)
: Conv2D에서 차이 없는 듯? 정확히

3. ```flow_from_directory```   
: return (x,y)   
x : (batch_size, target_size, channels) 크기의 이미지  
y : labels
   
### 2021.07.31
1. OOM (Out Of Memory)   
: 학습시키지 않고.. 근데 compile 생략했는데 compile과 train 차이,,?    
   가중치 로드하면 안 뜰 줄 알았는데 여전히 OOM 뜸

2. 1x1 합성곱의 의미? 무의미하다고 생각했음   
: 이미지 크기는 변함이 없는 게 맞음, 채널 수를 줄여서 파라미터를 줄여주는 효과

3. ResNet의 Residual : shortcut, 기울기 소실 문제를 해결하는 방식의 수학적인 부분이 잘 이해되지 않음

4. MaxPooling layer 를 거치면 spatial information 정보가 손실됨

### 2021.08.01
1. 공간 피라미드 풀링, R-CNN, Fast R-CNN, Faster R-CNN 잘 이해되지 않았음... 논문을 읽어봐?

2. 완전연결층의 한계 : 고정된 크기의 입력만 받아들이며, 완전연결층을 거친 후에는 위치 정보가 사라진다. 