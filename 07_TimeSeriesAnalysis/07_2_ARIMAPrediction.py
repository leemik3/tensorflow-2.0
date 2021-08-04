"""
chapter7. 시계열 분석
    7.2 AR, MA, ARMA, ARIMA

    ARIMA 모델을 사용하여 prediction
"""

'''
2. ARIMA 모델을 사용한 예측
'''

# statsmodels 라이브러리를 이용한 sales 데이터셋에 대한 예측
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):  # 시간을 표현하는 함수 정의
    return datetime.strptime('199'+x, "%Y-%m")

series = read_csv("../data/chap7/data/sales.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
X = np.nan_to_num(X)  # nan을 0으로 변경
size = int(len(X) * 0.66)  # train과 test를 66:34 비율로 나누겠다구~
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]  # train 데이터를 history에
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)  # 이게 그 arima 수식의 변수 값들을 찾는 부분인 거겠지?
    output = model_fit.forecast()  # forecast 메소드가 예측 수행
    yhat = output[0]  # yhat, 예측 결과 임시 저장소
    predictions.append(yhat)  # 예측 결과 쌓기
    obs = test[t]  # 실제 test 값
    history.append(obs)
    print('predicted=%f, expected=%f' %(yhat, obs) )

error = mean_squared_error(test, predictions)
print('Test MSE : %.3f' % error)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()