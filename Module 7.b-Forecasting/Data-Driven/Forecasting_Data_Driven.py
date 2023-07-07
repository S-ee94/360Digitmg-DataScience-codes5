'''#CRISP-ML(Q)
Business Problem: CocaCola management is reactive in understanding the sales patterns.
They are missing a chance to proactively take actions to meet their sales targets.

Business Objective: Maximize Revenue
Business Constraints: Minimize the gut feel based decisions

Success Criteria: 
    Business: Increase the revenue by at least 20%
    ML: Achieve an accuracy of at least 85%
    Economic: Achieve an increase in revenue by at least $200K

Data Understanding:
    Quaterly sales data from Q1 '86 until Q2 '96. In total we have 42 quarters of data. 
    We have two columns
    Column 1: Quarter
    Column 2: Sales (Target / Output) '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing
from sqlalchemy import create_engine

user = 'root' # user name
pw = 'Seemscrazy1994#' # password
db = 'cola_db' # database

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df = pd.read_excel(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Data-Driven\CocaCola_Sales_Rawdata.xlsx")

# dumping data into database 
df.to_sql('cocacola', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from cocacola'

cocacola = pd.read_sql_query(sql, con = engine)

cocacola.Sales.plot() # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = cocacola.head(38)
Test = cocacola.tail(4)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred, actual):
    temp = np.abs((pred - actual)/actual)*100
    return np.mean(temp)

# Moving Average for the time series
mv_pred = cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.Sales)

# Plot with Moving Averages
cocacola.Sales.plot(label = "actual")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
tsa_plots.plot_pacf(cocacola.Sales, lags = 4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
ses = MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
hw = MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
hwe = MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
hwe_w = MAPE(pred_hwe_mul_add, Test.Sales) 

# comparing all mape's
di = pd.Series({'Simple Exponential Method':ses, 'Holt method ':hw, 'hw_additive seasonality and additive trend':hwe, 'hw_multiplicative seasonality and additive trend':hwe_w})
mape = pd.DataFrame(di, columns=['mape'])
mape

# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# The models and results instances all have a save and load method, so you don't need to use the pickle module directly.
# to save model
hwe_model_add_add.save("model.pickle")

import os
os.getcwd()

# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Data-Driven\Newdata_CocaCola_Sales.xlsx")

newdata_pred = model.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred

fig, ax = plt.subplots()
ax.plot(new_data.Sales, '-b', label = 'Actual Value')
ax.plot(newdata_pred, '-r', label = 'Predicted value')
ax.legend();
plt.show()

##################################################