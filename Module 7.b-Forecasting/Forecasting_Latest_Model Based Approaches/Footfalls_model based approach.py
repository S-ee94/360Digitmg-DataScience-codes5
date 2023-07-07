'''# CRISP-ML(Q)
Business Problem: Walmart is not aware of the planning because they are unware of the number of customers who will visit their stores.
Business Objective: Maximize Customer Satisfication
Business Constraints: Minimize the number of customer service agents

Success Criteria: 
    Business: Increase the number of footfalls by at least 20%
    ML: Achieve an accuracy of at least 85%
    Economic: Achieve an increase in revenue by at least $200K

Data Understanding:
    Monthly data from Jan '91 until Mar '04. In total we have 159 months of data. 
    We have two columns
    Column 1: Date
    Column 2: Footfalls (Target / Output) '''

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine 

user = 'root' # user name
pw = 'Seemscrazy1994#' # password
db = 'wall_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Forecasting_Latest_Model Based Approaches\Walmart Footfalls Raw.csv")

# dumping data into database 
df.to_sql('walmart', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# loading data from database
sql = 'select * from walmart'

Walmart = pd.read_sql_query(sql, con = engine )

print(Walmart)


# Data Pre-processing

Walmart["t"] = np.arange(1, 160) # Linear Trend is captured
Walmart["t_square"] = Walmart["t"] * Walmart["t"] # Quadratic trend or polynomial with '2' degrees trend is captured
Walmart["log_footfalls"] = np.log(Walmart["Footfalls"]) # Exponential trend is captured
Walmart.columns

Walmart.info()

p = Walmart["Month"][0]
p[0:3]

Walmart['months'] = 0

for i in range(159):
    p = Walmart["Month"][i]
    Walmart['months'][i] = p[0:3]

    
month_dummies = pd.DataFrame(pd.get_dummies(Walmart['months']))
Walmart1 = pd.concat([Walmart, month_dummies], axis = 1)
Walmart1 = Walmart1.drop(columns = "months")

# Visualization - Time plot
Walmart1.Footfalls.plot()

# Data Partition
Train = Walmart1.head(147)
Test = Walmart1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1, 13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Footfalls ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_footfalls ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Footfalls ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad

################### Additive Seasonality ########################

add_sea = smf.ols('Footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't', 't_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_sea_linear = smf.ols('log_footfalls ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_Mult_sea_linear = pd.Series(Mul_sea_linear.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't']]))
rmse_Mult_sea_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Mult_sea_linear)))**2))
rmse_Mult_sea_linear

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_Mult_sea", "rmse_add_sea_quad", "rmse_Mult_sea_linear"]), "RMSE_Values":pd.Series([rmse_linear, rmse_Exp, rmse_Quad, rmse_add_sea, rmse_Mult_sea, rmse_add_sea_quad, rmse_Mult_sea_linear])}
table_rmse = pd.DataFrame(data)
table_rmse

# 'rmse_add_sea_quad' has the least RMSE value among the models prepared so far. Use these features and build forecasting model using entire data
model_full = smf.ols('Footfalls ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Walmart1).fit()

predict_data = pd.read_excel(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7.b-Forecasting\Forecasting_Latest_Model Based Approaches\Predict_new.xlsx")

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Footfalls"] = pd.Series(pred_new)


# The models and results have save and load method, so you don't need to use the pickle module directly.
# to save model
model_full.save("model.pickle")

import os
os.getcwd()

# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# RESIDUALS MIGHT HAVE ADDITIONAL INFORMATION!

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = Walmart1.Footfalls - model.predict(Walmart1)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of Y with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags = 12)

# Alternative approach for ACF plot is explained in next 2 lines
# from pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()
                          
# AR Autoregressive model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags = [1])
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)

pred_res = model_fit.predict(start = len(full_res), end = len(full_res) + len(predict_data) - 1, dynamic = False)
pred_res.reset_index(drop = True, inplace = True)

# The Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred


