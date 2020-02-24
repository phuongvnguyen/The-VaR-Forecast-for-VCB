#!/usr/bin/env python
# coding: utf-8

# $$\large \color{green}{\textbf{The Value-At-Risk Forecasting for Vietcomebank}}$$ 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# 
# This computer program was written by Phuong V. Nguyen, based on the $\textbf{Anacoda 1.9.7}$ and $\textbf{Python 3.7}$.
# 
# $$\text{1. Issue}$$
# 
# This project is to forecast the Value-At-Risk of the Vietcombank stock
# 
# $$\text{2. Methodology}$$
# 
# The GARCH model specification
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%. It is worth noting that there are a number of methods to calculate this qualtile, such as the parametric (or the variance–covariance approach), the Historical Simulation.
# 
# 
# $$\text{3. Dataset}$$ 
# 
# One can download the dataset used to replicate my project at my Repositories on the Github site below
# 
# https://github.com/phuongvnguyen/The-VaR-Forecast-for-VCB
# 
# Or update it at
# 
# https://www.vndirect.com.vn/portal/thong-ke-thi-truong-chung-khoan/lich-su-gia.shtml
# 
# 
# # Preparing Problem
# 
# ##  Loading Libraries

# In[1]:


import warnings
import itertools
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from arch import arch_model
from arch.univariate import GARCH


# ## Defining some varibales for printing the result

# In[2]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# ##  Loading Dataset

# In[56]:


data = pd.read_excel("VCBdata.xlsx")


# # Data Exploration and Preration
# 
# ## Data exploration

# In[57]:


data.head(5)


# ## Computing returns
# ### Picking up the close prices

# In[58]:


closePrice = data[['DATE','CLOSE']]
closePrice.head(5)


# ### Computing the daily returns

# In[59]:


closePrice['Return'] = closePrice['CLOSE'].pct_change()
closePrice.head()


# In[60]:


daily_return=closePrice[['DATE','Return']]
daily_return.head()


# ### Reseting index

# In[61]:


daily_return =daily_return.set_index('DATE')
daily_return.head()


# In[62]:


daily_return = 100 * daily_return.dropna()
daily_return.head()


# In[63]:


daily_return.index


# ### Plotting returns

# In[65]:


sns.set()
fig=plt.figure(figsize=(12,6))
plt.plot(daily_return.Return[:'2019'],LineWidth=1)
plt.autoscale(enable=True,axis='both',tight=True)
#plt.grid(linestyle=':',which='both',linewidth=2)
fig.suptitle('The Log Daily Returns of the Vietcombank Stock', fontsize=15,fontweight='bold'
             ,color='b')
plt.title('01/07/2009- 31/12/2019',fontsize=13,fontweight='bold',
          color='b')
plt.ylabel('Return (%)',fontsize=10)
plt.xlabel('Source: The Daily Close Price-based Calculations',fontsize=10,fontweight='normal',color='k')


# # Modelling GARCH model
# 
# $$\text{Mean equation:}$$
# $$r_{t}=\mu + \epsilon_{t}$$
# 
# $$\text{Volatility equation:}$$
# $$\sigma^{2}_{t}= \omega + \alpha \epsilon^{2}_{t} + \beta\sigma^{2}_{t-1}$$
# 
# $$\text{Volatility equation:}$$
# 
# $$\epsilon_{t}= \sigma_{t} e_{t}$$
# 
# $$e_{t} \sim N(0,1)$$
# 

# In[66]:


for row in daily_return.index: 
    print(row)


# In[67]:


#garch = arch_model(daily_return,mean='AR',lags=5,
 #                  vol='GARCH',dist='studentst',
  #              p=1, o=0, q=1)
garch = arch_model(daily_return,vol='Garch', p=1, o=0, q=1, dist='skewt')
results_garch = garch.fit(last_obs='2018-12-28', update_freq=1,disp='on')
print(results_garch.summary())


# # Estimating the VaR
# 
# we use the model to estimate the VaR
# 
# 
# Value-at-Risk (VaR) forecasts from GARCH models depend on the conditional mean, the conditional volatility and the quantile of the standardized residuals,
# 
# $$\text{VaR}_{t+1|t}=\mu_{t+1} -\sigma_{t+1|t}q_{\alpha} $$
# 
# 
# where $q_{\alpha}$ is the $\alpha$ quantile of the standardized residuals, e.g., 5%.
# 
# ## Computing the quantiles
# 
# ### The Filtered Historical Simulation
# 
# The quantiles, $q_{\alpha}$, can be computed using the Filtered Historical Simulation below.
# 
# #### Computing the standardized residuals
# 
# It is worth noting that the standardized residuals computed by conditional volatility as follows.

# In[68]:


std_garch = (daily_return.Return[:'2018'] - results_garch.params['mu']) / results_garch.conditional_volatility
std_garch = std_garch.dropna()
std_garch.head(5)


# #### Computing the Quantiles
# 
# At the probabilities of 1% and 5%

# In[69]:


FHS_quantiles_VaRgarch = std_garch.quantile([.01, .05])
print(Bold+'The quantiles at the probabilities of 1% and 5% are as follows'+End)
print(FHS_quantiles_VaRgarch)


# #### Computing the conditional mean and volatilitie

# In[70]:


FHS_forecasts_VaRgarch = results_garch.forecast(start='2019-01-02')
FHS_cond_mean_VaRgarch = FHS_forecasts_VaRgarch.mean['2019':]
FHS_cond_var_VaRgarch = FHS_forecasts_VaRgarch.variance['2019':]


# #### Computing the Value-At-Risk (VaR)

# In[75]:


FHS_value_at_risk = -FHS_cond_mean_VaRgarch.values - np.sqrt(FHS_cond_var_VaRgarch).values * FHS_quantiles_VaRgarch[None, :]

FHS_value_at_risk = pd.DataFrame(
    FHS_value_at_risk, columns=['1%', '5%'], index=FHS_cond_var_VaRgarch.index)

FHS_value_at_risk.head(5)


# #### Visualizing the VaR vs actual values
# ##### Picking actual data

# In[76]:


rets_2019= daily_return['2019':].copy()
rets_2019.name = 'Return'
rets_2019.head(5)


# ##### Plotting

# In[78]:


fig=plt.figure(figsize=(12,5))
plt.plot(FHS_value_at_risk['1%'] ,LineWidth=2,
         linestyle='--',label='VaR returns at 1%')
plt.plot(FHS_value_at_risk['5%'] ,LineWidth=2,
         linestyle=':',label='VaR returns at 5%')
plt.plot(rets_2019['Return'] ,LineWidth=2,
         linestyle='-',label='Actual return')
plt.suptitle('The Daily FHS-based Value-At-Risk (VaR) Measurements of the Vietcombank stock', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('01/02/2019-31/12/2019',fontsize=10,
          fontweight='bold',color='b')
plt.autoscale(enable=True,axis='both',tight=True)
plt.legend()


# ### Parametric Method
# 
# #### Computing the quantile

# In[81]:


param_quantiles_VaRgarch = garch.distribution.ppf([0.01, 0.05], results_garch.params[-2:])
print(Bold+'The quantiles at the probabilities of 1% and 5% are as follows'+End)
print(param_quantiles_VaRgarch)


# #### Computing the conditional mean and volatilitie

# In[82]:


param_forecasts_VaRgarch = results_garch.forecast(start='2019-01-02')
param_cond_mean_VaRgarch = param_forecasts_VaRgarch.mean['2019':]
param_cond_var_VaRgarch = param_forecasts_VaRgarch.variance['2019':]


# #### Computing the Value-At-Risk (VaR)

# In[83]:


param_value_at_risk = -param_cond_mean_VaRgarch.values - np.sqrt(param_cond_var_VaRgarch).values * param_quantiles_VaRgarch[None, :]

param_value_at_risk = pd.DataFrame(
    param_value_at_risk, columns=['1%', '5%'], index=param_cond_var_VaRgarch.index)

param_value_at_risk.head(5)


# #### Visualizing the VaR vs actual values
# 

# In[84]:


fig=plt.figure(figsize=(12,5))
plt.plot(param_value_at_risk['1%'] ,LineWidth=2,
         linestyle='--',label='VaR returns at 1%')
plt.plot(param_value_at_risk['5%'] ,LineWidth=2,
         linestyle=':',label='VaR returns at 5%')
plt.plot(rets_2019['Return'] ,LineWidth=2,
         linestyle='-',label='Actual return')
plt.suptitle('The Daily Parametric-based Value-At-Risk (VaR) Measurements of the Vietcombank stock', 
             fontsize=15,fontweight='bold',
            color='b')
plt.title('01/02/2019-31/12/2019',fontsize=10,
          fontweight='bold',color='b')
plt.autoscale(enable=True,axis='both',tight=True)
plt.legend()


# In[ ]:




