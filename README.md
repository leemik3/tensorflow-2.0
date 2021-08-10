# tensorflow-2.0
할 거 없을 때 check 빼고 wiki로 옮기기

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
- y가 연속형 변수일 때 주로 사용 
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
- 데이터 양 늘리기
- 드롭아웃
    - 신경망 학습 시에만 사용하고, 예측 시에는 사용하지 않는 것이 일반적
    - 서로 다른 신경망들을 앙상블하여 사용하는 것 같은 효과
- 모델의 복잡도 줄이기
    - 모델의 수용력 (capacity) : 모델에 있는 매개변수들의 수
- 가중치 규제 (Regularization) * 정규화가 아님
    - L1 규제 / L1 Norm   
        - 가중치 w들의 절대값 합계를 비용 함수에 추가
        - 어떤 특성들이 모델에 영향을 주고 있는지 정확히 판단할 때 유용
    - L2 규제 / L2 Norm (weight decay)
        - 가중치 w들의 제곱합을 비용 함수에 추가
        - 어떤 특성이 모델에 영향을 주고 있는지 판단하는 것이 필요없다면 경험적으로 L1 규제보다 더 잘 작동함
    - 비용 함수를 최소화하려면, 가중치 값이 작아져야 한다.
      

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
### Batch, Iteration, Epoch
- **batch size**  
mini batch의 데이터 수, 가중치 1회 갱신 기준   
- **iteration**  
한 epoch를 진행하기 위해서 이뤄지는 가중치 갱신 횟수   
- **epoch**  
전체 데이터셋에 대해서 1회 학습 기준   
- batch size * iteration = 전체 데이터 수   

---

# 합성곱 신경망

### Conv1D, Conv2D, Conv3D 데이터 형태 정리



### height, width, channel, length, depth 관계 정리
ex) Conv2D(~, input_shape=(<행>,<열>,<채널 개수>))

filter 개수 ≠ filter channel 수

---

### 기울기 소멸 / 소실 문제 (Vanishing Gradient Problem) & 폭주 문제
오차 정보를 역전파시키는 과정에서 기울기가 급격히 0에 가까워져 학습이 안 되는 현상  
- 원인
    - 내부 공변량 변화 (internal covariance shift) : 네트워크의 각 층마다 활성화 함수가 적용되면서 입력 값들의 분포가 계속  바뀌는 현상
- 해결 방법
    - 손실 함수로 ReLU 사용
    - 초깃값 튜닝
    - 학습률 조정
    - 배치 정규화
    - gradient clipping


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
### AR, MA, ARMA, ARIMA
- AR 모델 (Auto Regression, 자기 회귀)
    - 이전 관측 값이 이후 관측 값에 영향을 준다는 아이디어
    - 이전 데이터의 **상태**에서 현재 데이터의 상태 추론   
- MA 모델 (Moving Average, 이동 평균)
    - 트렌드(평균이나 시계열 그래프의 y값)가 변화하는 상황
    - 이전 데이터의 **오차**에서 현재 데이터의 상태 추론
- ARMA 모델 (AutoRegressive Moving Average, 자동 회귀 이동 평균)
    - 윈도우 크기만큼 슬라이딩(moving)   
- ARIMA 모델 (AutorRegressive Integrated Moving Average, 자동 회귀 누적 이동 평균)
    - ARMA와 달리 과거 데이터의 선형 관계뿐만 아니라 추세(cointegration)까지 고려함
---
### RNN (Recurrent Neural Network)
- 시간적으로 연속성이 있는 데이터를 처리하려고 고안된 인공 신경망
- 이전 은닉층이 현재 은닉층의 입력이 되면서 반복되는 순환 구조를 갖는다
- 기존 네트워크와 다른 점은 '기억을 갖는다'는 것

- 활용 분야 
    - 자연어 처리 : 음성 인식, 단어 의미 판단 및 대화 등 처리
    - 시계열 데이터 처리
---
### RNN의 유형
1. 일대일
    - 순환이 없기 떄문에 RNN이라고 말하기 어려움
    - EX) 순방향 네트워크   
2. 일대다
    - 입력이 하나, 출력이 다수
    - EX) image captioning (입력 : 이미지, 출력 : 이미지에 대한 설명 = 문장)   
3. 다대일
    - 입력이 다수, 출력이 하나
    - EX) 감정 분석기 (입력 : 문장, 출력 : 긍/부정)   
4. 다대다
    - 입력이 다수, 출력이 다수
    - EX) 자동 번역기   
5. 동기화 다대다
    - 입력이 다수, 출력이 다수
    - EX) 문장에서 다음에 나올 단어를 예측하는 언어 모델, 프레임 수준의 비디오 분류기
---
### RNN 계층과 셀
RNN 계층 : 입력된 배치 순서열을 모두 처리  
RNN 셀 : 하나의 단계 (timestamp)만 처리

---
### RNN의 역전파 : BPTT
모든 단계마다 처음부터 끝까지 역전파  
- 오차가 멀리 전파될 때 : 기울기 소멸 문제 발생 - 해결방안  
    - truncated BPTT (일정 시점까지만 오류 역전파)  
    - LSTM  
    - GRU
---
### LSTM
- RNN의 기울기 소멸 문제 방지
- 망각 게이트, 입력 게이트, 출력 게이트를 은닉층의 각 뉴런에 추가
1. **망각 게이트 (forget gate)**
- 과거 정보를 얼마나 기억할지 결정
- 과거 정보와 현재 데이터를 입력 받아 시그모이드를 취한 후 그 값을 과거 정보에 곱함 (시그모이드 출력값에 따라 과거 정보를 버리거나 보존)
2. **입력 게이트(input gate)**
- 현재 정보를 기억하기 위해 만들어짐
- 현재 정보에 대한 보존량 결정
3. **출력 게이트(output gate)**
- 과거 정보와 현재 데이터를 사용하여 뉴런의 출력을 결정
---
### GRU (Gated Recurrent Unit)
- 망각 게이트와 입력 게이트를 합친 것
- 출력 게이트 없음 : 전체 상태 벡터가 매 단계마다 출력됨
- 게이트 컨트롤러 : 망각 게이트와 입력 게이트를 모두 제어
    - 1을 출력 : 망각 게이트 열리고, 입력 게이트 닫힘
    - 0을 출력 : 망각 게이트 닫히고, 입력 게이트 열림
    
1. **망각 게이트 (reset gate)**
- 과거 정보를 적당히 초기화 시키려는 목적
2. **업데이트 게이트(update gate)**
- 과거와 현재 정보의 최신화 비율 결정
3. **후보군**
- 현시점의 정보에 대한 후보군 계산
4. **은닉층**
- 업데이트 결과와 후보군 결과를 결합하여 현시점의 은닉층 계산
---
### 양방향 RNN (Bidirectional RNN)
- 이전 시점 뿐만 아니라 미래 시점의 데이터도 함께 활용
- 하나의 출력값을 예측하는 데 메모리 셀 두개를 사용
    - 첫 번째 메모리 셀 : **이전** 시점의 은닉 상태 -> 현재의 은닉 상태 계산
    - 두 번째 메모리 셀 : **다음** 시점의 은닉 상태 -> 현재의 은닉 상태 계산
---
# Attention Mechanism (어텐션 메커니즘)

    

---
# 성능 최적화 (Optimization) 방법
### 데이터를 사용
1. 최대한 많은 데이터 수집
2. 데이터 생성  
: EX) ImageDataGenerator
3. 데이터 범위(scale) 조정하기  
시그모이드 : 0 ~ 1  
tanh : -1 ~ 1  
정규화, 규제화, 표준화
### 알고리즘을 이용
서로 다른 알고리즘들은 선택하여 훈련시켜 보고 성능이 가장 좋은 모델을 선택하기
### 알고리즘 튜닝을 위한 성능 최적화
가장 많은 시간이 소요되는 부분
- 진단   
    - overfitting (훈련 성능 >> 검증 성능) -> 규제화  
    - underfitting (훈련 성능, 검증 성능 ↓) -> 네트워크 구조 변경, 에포크 수 조정  
    - 훈련 성능이 검증을 넘어서는 변곡점에서 조기 종료 고려
- 가중치 
    - 초깃값 : 작은 난수
    - 초깃값 : 오토인코더 같은 비지도 학습을 이용하여 사전 훈련(가중치 정보를 얻기 위함)을 진행한 후 지도 학습을 진행하는 것도 방법
- 학습률
    - 모델의 네트워크 구성에 따라 다르기 때문에 매우 크거나 작은 임의의 난수를 선택하고 조금씩 변경해야 함.
    - 네트워크 계층 ↑ 학습률 ↑, 네트워크 계층 ↓ 학습률 ↓
- 활성화 함수
    - **활성화 함수**를 변경할 땐 **손실 함수**와 함께 변경해야 하는 경우가 많으므로 신중해야 함
    - 일반적) 활성화 함수 : 시그모이드, tanh / 출력층에서는 softmax나 sigmoid 많이 사용함
- 배치와 에포크
    - 최근 트렌드 : 큰 에포크와 작은 배치
    - 다양한 테스트 진행해보기
- 옵티마이저 및 손실 함수
    - 일반적) 옵티마이저 : 확률적 경사 하강법
    - Adam, RMSProp
    - 다양한 테스트 진행해보기
- 네트워크 구성 (네트워크 토폴로지 / topology)
    - 너비와 깊이 조절

### 앙상블을 이용
간단히 모델을 두 개 이상 섞어서 사용하기
### 하드웨어(GPU)를 이용
Windows 환경 : GPU용 텐서플로를 설치하려면 **CUDA**와 **cuDNN** 설치
- CUDA (Computed Unified Device Architecture) : NVIDIA에서 개발한 GPU 개발 툴
- GPU 설치 확인 : ```nvidia -smi```
### 하이퍼파라미터를 이용
- **배치 정규화**
    - 매 단계마다 활성화 함수를 거치면서 데이터셋 분포가 일정해지기 때문에 속도를 향상시킬 수 있다.
    - 단점
        - 배치 크기가 작을 때는 정규화 값이 기존 값과 다른 방향으로 훈련될 수 있다. 분산이 0이면 정규화 자체가 안 됨
        - RNN은 네트워크 계층 별로 미니 정규화를 적용해야 해서 모델이 더 복잡해지고 비효율적일 수 있다.
    
- **드롭아웃 (Dropout)**
    - 훈련 시간이 길어지지만, 모델 성능 향상에 자주 쓰는 방법

- **조기종료 (Early Stopping)**
    - 검증에 대한 손실이 증가하는 시점에서 훈련 멈추도록 조정

---
### CPU와 GPU
- 개별적 코어 속도 : CPU > GPU
- **CPU** : 명령어가 입력되는 순서대로 데이터를 처리하는 직렬 처리 방식  
산술논리장치(ALU) : 연산을 담당  
컨트롤 (control) : 명령어를 해석하고 실행  
캐시 (cache) : 데이터를 담아 둠
    - 순차적 연산에 적합
- **GPU** (Graphics Processing Units): 서로 다른 명령어를 동시에 병렬 처리  
ALU 개수가 많아지고, 캐시 메모리 비중이 낮아짐 (데이터를 담아두지 않고 명령어 많이 처리)
    - 역전파처럼 복잡한 미적분은 병렬 연산을 해야 속도가 빨라짐
---
### 정규화, 규제화, 표준화
- **정규화 (normalization)**
    - 데이터 범위를 사용자가 원하는 범위로 제한하는 것
    - 특성 스케일링 : 각 특성 범위를 조정한다는 의미
    - ```MinMaxScaler()```
- **규제화 (regularization)**
    - 모델 복잡도를 줄이기 위해 제약을 두는 방법
    - 데이터가 네트워크에 들어가기 전에 **필터**가 적용된 것이라고 생각하면 됨
    - 규제를 이용하여 모델 복잡도를 줄이는 방법 : 드롭아웃, 조기 종료
- **표준화 (standardization)**
    - 평균은 0, 표준편차는 1 인 데이터로 만들기
    - 다른 표현 : 표준화 스칼라 (standard scalar), z-스코어 정규화 (z-score normalization)
---
### Unit, input_shape 정리
```Dense(64, input_shape=(4,), activation='relu')```

---
# Clustering
- K-means clustering
- Gaussian Mixture Model
- 자기 조직화 지도 (SOM: Self-Organizing Map)
---
### K-means clustering


---
# Check
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

### 2021.08.05
1. 가정설정문 assert : 뒤의 조건이 거짓일 경우 에러 발생시킴
2. ```tf.keras.preprocessing.sequence.pad_sequences```  
   : Transforms a **list** (길이 : num_samples) of sequences (lists of integers)   
   into a **2D Numpy array** of shape (num_samples, num_timesteps)
3. ```tf.unstack()```
4. Tensorslicedataset, Batchdataset, onehotencoding 등 적재적소 데이터 전처리에 대한 이해
5. __call__ : [매직 메소드](https://github.com/leemik3/python/wiki/%ED%81%B4%EB%9E%98%EC%8A%A4(class))
6. '07_3_RNN_Cell.py' 클래스 부분 이해 안 감
7. ```VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])```  
: 경고 무시했음, 동작은 됨
   
### 2021.08.06
1. ```Dense(64, input_shape=(4,), activation='relu')```  
- 유닛이 64개인건 이해 됨.
- (입력층이) (4,0) 형태를 가진다고 책에 써있음. 처음에는 4가 batch size, 데이터 크기인가 싶었는데, 데이터 1개가 (4,0) 형태,, 컬럼 개수인가? 싶었음... 계속 헷갈림 이게

### 2021.08.07
1. ```tfds.load``` : ProfetchDataset,, for 문으로 출력하면 내용 볼 수 있음
2. ```padded_batch()``` : 배치에서 가장 긴 문자열의 길이를 기준으로 시퀀스를 0으로 채움
3. ```shuffle``` : ??

### 2021.08.09
1. ```tf.keras.callbacks.ModelCheckpoint``` 콜백 함수 : 훈련 중간 / 마지막에 체크포인트 사용
2. ```sequence.pad_sequences(x_train, maxlen=maxlen)``` 0으로 시퀀스를 채움

### 2021.08.10
1. ```glob.glob('<경로>'')``` : 해당 경로에 잇는 모든 폴더 및 파일을 **리스트**로 반환
2. 