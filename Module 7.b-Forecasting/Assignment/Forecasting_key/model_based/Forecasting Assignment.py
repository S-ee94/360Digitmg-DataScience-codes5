import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from urllib.parse import quote 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
import calendar
import statsmodels.api as sm
import pmdarima as pm

# connecting sql with python
user_name = 'root'
database = 'model_db'
pw = 'Seemscrazy1994#'

engine = create_engine(f'mysql+pymysql://{user_name}:%s@localhost:3306/{database}' % quote(f'{pw}'))
df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Assignment\Forecasting_key\Data_Driven\solarpower_cumuldaybyday2.csv")
# dumping data into database 
df.to_sql('solarpower_cumuldaybyday2'.lower(), con = engine, if_exists = 'replace', chunksize = 1000, index = False)
# loading data from database
sql = 'select * from solarpower_cumuldaybyday2'
solarpower = pd.read_sql_query(sql, con = engine )


solarpower = solarpower.rename(columns={"date":"Month"})
solarpower.insert(2,'Power',0.1)
for i in range(len(solarpower)):
    if i==0:
        solarpower['Power'][i]= 0.1
    else:
        solarpower['Power'][i]= solarpower['cum_power'][i]-solarpower['cum_power'][i-1]
        
        
solarpower= solarpower.drop('cum_power',axis=1)
solarpower

solarpower.Month = pd.to_datetime(solarpower.Month)
Solar1 = solarpower.resample('M', on='Month').sum()
Solar1.head()

month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

dates = pd.date_range(start='2011-01-01', freq='MS',periods=len(Solar1))
Solar1 ['Months'] = dates.month
Solar1 ['Year'] = dates.year
Solar1

Solar1 ['Months'] = Solar1 ['Months'].apply(lambda x: calendar.month_abbr[x])
Solar1 = Solar1[['Months','Year','Power']]
Solar1
# Pre processing
Solar1["t"] = np.arange(1,97)

Solar1["t_square"] = Solar1["t"] * Solar1["t"]
Solar1["log_power"] = np.log(Solar1["Power"])
Solar1.columns
Solar1
dummy= pd.DataFrame(pd.get_dummies(Solar1['Months']))

Solar1=pd.concat((Solar1,dummy),axis=1)
train = Solar1.head(84)
test = Solar1.tail(12)


#linear model
linear= smf.ols('Power~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Power'])-np.array(predlin))**2))
rmselin

#quadratic model
quad=smf.ols('Power~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Power'])-np.array(predquad))**2))
rmsequad

#exponential model
expo=smf.ols('log_power~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Power'])-np.array(np.exp(predexp)))**2))
rmseexpo

#additive seasonality
additive= smf.ols('Power~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
predadd= pd.Series(additive.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Power'])-np.array(predadd))**2))
rmseadd

#additive seasonality with linear trend
addlinear= smf.ols('Power~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
predmul=pd.Series(addlinear.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t']]))
predadd
rmseaddlinear=np.sqrt(np.mean((np.array(test['Power'])-np.array(predadd))**2))
rmseaddlinear

#additive seasonality with quadratic trend
addquad=smf.ols('Power~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
predmuladd= pd.Series(addquad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_square']]))
rmseaddquad=np.sqrt(np.mean((np.array(test['Power'])-np.array(predmuladd))**2))
rmseaddquad

#multiplicative seasonality
mulsea=smf.ols('log_power~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmsemul= np.sqrt(np.mean((np.array(test['Power'])-np.array(np.exp(predmul)))**2))
rmsemul

#multiplicative seasonality with linear trend
mullin= smf.ols('log_power~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
predmulquad= pd.Series(mullin.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t']]))
rmsemulin=np.sqrt(np.mean((np.array(test['Power'])-np.array(np.exp(predmulquad)))**2))
rmsemulin

#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_power~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_square']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Power'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad

#tabulating the rmse values
data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
Rmse=pd.DataFrame(data)
Rmse

#final model with least rmse value
Predict_Solar = Solar1.copy()
final = smf.ols('Power ~ t',data = Solar1).fit()
pred = pd.Series(final.predict(Predict_Solar))
pred

Predict_Solar['prediction']=pred
residuals=pd.DataFrame(np.array(Solar1["Power"]-np.array(pred)))

sm.graphics.tsa.plot_acf(residuals.values.squeeze(), lags=12)
sm.graphics.tsa.plot_pacf(Solar1.Power.values.squeeze(), lags=12)
# ARIMA with AR = 4, MA = 6

model1 = ARIMA(Solar1["Power"], order = (2,0,2))
res1 = model1.fit()
print(res1.summary())

pas=Solar1["Power"]
model_Arima=ARIMA(residuals,order=(2,0,2)).fit()
forecasterrors=model_Arima.forecast(steps=12)
forecasterrors
# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user


ar_model = pm.auto_arima(Solar1["Power"], start_p = 0, start_q = 0,
                      max_p = 16, max_q = 16, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)

model1 = ARIMA(Solar1["Power"], order = (2,1,3))
res1 = model1.fit()
print(res1.summary())
test = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Assignment\Forecasting_key\Data_Driven\Solar_test.csv", index_col = 0)
test.index = pd.to_datetime(test.index)
test.head()
test.index[0]
predict = res1.predict(start=test.index[0], end=test.index[-1])
predict

# Future forecasts
start = pd.to_datetime('2019-01-01')
end = pd.to_datetime('2019-12-31')
forecast = res1.predict(start=start, end=end)
forecast
res1.save("model_Arima.pickle")

#to load model

from statsmodels.regression.linear_model import OLSResults
model_Arima = OLSResults.load("model_Arima.pickle")
