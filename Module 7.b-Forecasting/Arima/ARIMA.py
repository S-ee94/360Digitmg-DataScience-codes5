import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot 
from sqlalchemy import create_engine

user = 'root'
password = 'Seemscrazy1994#'
db = 'arima_db'

engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Arima\Walmart Footfalls Raw.csv")

# dumping data into database
df.to_sql('walmart', con = engine, if_exists = 'replace', index = False, chunksize = 1000)

# loading data from database
sql = 'select * from walmart'
Walmart = pd.read_sql_query(sql, con = engine)


# Data Partition
Train = Walmart.head(147)
Test = Walmart.tail(12)

Test.to_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Arima\test_arima.csv")
import os
os.getcwd()

df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Arima\test_arima.csv", index_col = 0)

tsa_plots.plot_acf(Walmart.Footfalls, lags = 12)
tsa_plots.plot_pacf(Walmart.Footfalls, lags = 12)


# ARIMA with AR = 12, MA = 6
model1 = ARIMA(Train.Footfalls, order = (12, 1, 6))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Footfalls, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test, color = 'red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm 

help(pm.auto_arima)

ar_model = pm.auto_arima(Train.Footfalls, start_p = 0, start_q = 0,
                      max_p = 12, max_q = 12, # maximum p and q
                      m = 12,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = True,   # Seasonality
                      start_P = 0, trace = True,
                      error_action = 'warn', stepwise = True)


# Best Parameters ARIMA
# ARIMA with AR = 2, I = 1, MA = 0
model = ARIMA(Train.Footfalls, order = (2, 1, 0))
res = model.fit()
print(res.summary())


# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Footfalls, forecast_best))
print('Test RMSE: %.3f' % rmse_best)
# plot forecasts against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_best, color = 'red')
pyplot.show()


# checking both rmse of with and with out autoarima

print('Test RMSE with Auto-ARIMA: %.3f' % rmse_best)
print('Test RMSE with out Auto-ARIMA: %.3f' % rmse_test)
# saving model whose rmse is low
# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
res1.save("model.pickle")
# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Forecast for future 12 months
start_index = len(Walmart)
end_index = start_index + 11
forecast = model.predict(start = start_index, end = end_index)

print(forecast)
