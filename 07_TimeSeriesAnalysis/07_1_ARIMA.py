"""
chapter7. 시계열 분석
    7.2 AR, MA, ARMA, ARIMA

    ARIMA 모델 구현
"""

'''
1. ARIMA 모델

: AR (자기 회귀)와 MA (이동 평균) 모두를 고려하는 모형
ARIMA(order = (p, d, q) )
- p : 자기 회귀 차수 (AR 모형)
- d : 차분 차수
- q : 이동 평균 차수 (MA 모형)
> p+q < 2 또는 p*q = 0으로 많이 사용, ∵ 대부분의 시계열 자료에서는 하나의 경향만을 강하게 띄므로
'''

# ARIMA() 함수를 호출하여 sales 데이터셋에 대한 예측
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

def parser(x):  # 시간을 표현하는 함수 정의
    return datetime.strptime('199'+x, "%Y-%m")

series = read_csv("../data/chap7/data/sales.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)  # 디버그 정보 제공 비활성화
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())