import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing
from sqlalchemy import create_engine
from urllib.parse import quote

user = 'root' # user name
pw = 'Seemscrazy1994#' # password
db = 'datadriven_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Assignment\Forecasting_key\Data_Driven\solarpower_cumuldaybyday2.csv")

# dumping data into database 
df.to_sql('solarpower', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from solarpower'
solarpower = pd.read_sql_query(sql, con = engine )
solarpower
solarpower.insert(2,'Power',0.1)
for i in range(len(solarpower)):
    if i==0:
        solarpower['Power'][i]= 0.1
    else:
        solarpower['Power'][i]= solarpower['cum_power'][i]-solarpower['cum_power'][i-1]
        
        
solarpower= solarpower.drop('cum_power',axis=1)
solarpower = solarpower.rename(columns={"date":"Month"})
solarpower.Month = pd.to_datetime(solarpower.Month)
Solar1 = solarpower.resample('M', on='Month').sum()
Solar1


solarpower.Power.plot() # time series plot 

# Splitting the data into Train and Test data<br>
# Recent 4 time period values are Test data
Train = Solar1.head(84)
Test = Solar1.tail(12)
Test
Test.to_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Assignment\Forecasting_key\Data_Driven\Solar_test.csv", index = True)
# Creating a function to calculate the MAPE value for test data 
def MAPE(pred, actual):
    temp = np.abs((pred - actual)/actual)*100
    return np.mean(temp)

# Moving Average for the time series
mv_pred = Solar1["Power"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.Power)
# Plot with Moving Averages
Solar1.Power.plot(label = "actual")
for i in range(2, 9, 2):
    Solar1["Power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.<br>
# Time series decomposition plot using Moving Average

decompose_ts_add = seasonal_decompose(Solar1.Power, model = "additive", period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Solar1.Power, model = "multiplicative", period = 4)
decompose_ts_mul.plot()


# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(Solar1.Power, lags = 4)
tsa_plots.plot_pacf(Solar1.Power, lags = 4)

# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.
# PACF is a partial auto-correlation function. <br>
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
ses = MAPE(pred_ses, Test.Power) 
ses

# Holt method 
hw_model = Holt(Train["Power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
hw = MAPE(pred_hw, Test.Power) 
hw

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Power"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
hwe = MAPE(pred_hwe_add_add, Test.Power) 
hwe

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Power"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
hwe_w = MAPE(pred_hwe_mul_add, Test.Power) 
hwe_w

# comparing all mape's
di = pd.Series({'Simple Exponential Method':ses, 'Holt method ':hw, 'hw_additive seasonality and additive trend':hwe, 'hw_multiplicative seasonality and additive trend':hwe_w})
mape = pd.DataFrame(di, columns=['mape'])
mape

# Final Model on 100% Data
hw_model = Holt(Solar1["Power"]).fit()
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.<br>
# to save model
hw_model.save("model.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")
# Load the new data which includes the entry for future 4 values
new_data = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Assignment\Forecasting_key\Data_Driven\Solar_test.csv")
new_data.Month = pd.to_datetime(new_data.Month)
new_data = new_data.set_index('Month')
new_data
newdata_pred = model.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred


fig, ax = plt.subplots()
ax.plot(new_data.Power, '-b', label = 'Actual Value')
ax.plot(newdata_pred, '-r', label = 'Predicted value')
ax.legend();
plt.show()

start = pd.to_datetime('2019-01-01')
end = pd.to_datetime('2019-12-31')
# for future 12 months forecasting
newdata_pred = hw_model.predict(start = start, end = end)
newdata_pred
################################################