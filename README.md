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

