#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('PADM CW2 Dataset.csv')


# In[3]:


data = data.drop('Number', axis=1)


# In[4]:


data['observation_date'] = pd.to_datetime(data['observation_date'])


# In[5]:


data.set_index('observation_date', inplace=True)


# In[6]:


descriptive_stats = data.describe()
print(descriptive_stats)


# In[7]:


fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(data['S&P 500'])
axs[0, 0].set_title('S&P 500')
axs[0, 1].plot(data['Inflation Rate'])
axs[0, 1].set_title('Inflation Rate')
axs[1, 0].plot(data['GDP'])
axs[1, 0].set_title('GDP')
axs[1, 1].plot(data['Interest Rate'])
axs[1, 1].set_title('Interest Rate')
plt.tight_layout()
plt.show()


# In[9]:


#Logarithmic transformation

import numpy as np

data['ln_S&P500'] = np.log(data['S&P 500'])
data['ln_GDP'] = np.log(data['GDP'])


# In[11]:


#Augmented Dickey-Fuller (ADF) test for stationarity
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

print('ADF test results for ln_S&P500:')
adf_test(data['ln_S&P500'])
print('ADF test results for ln_GDP:')
adf_test(data['ln_GDP'])
print('ADF test results for Inflation Rate:')
adf_test(data['Inflation Rate'])
print('ADF test results for Interest Rate:')
adf_test(data['Interest Rate'])


# In[13]:


#Johansen cointegration test
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Perform Johansen cointegration test
result = coint_johansen(data[['ln_S&P500', 'ln_GDP', 'Inflation Rate', 'Interest Rate']], det_order=0, k_ar_diff=1)

print('Trace Statistic:')
print(result.lr1)
print('Critical Values (90%, 95%, 99%):')
print(result.cvt)
print('Eigenvalues:')
print(result.eig)

print('\nMaximum Eigenvalue Statistic:')
print(result.lr2)
print('Critical Values (90%, 95%, 99%):')
print(result.cvm)


# # error correction model (ECM)

# In[14]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# In[15]:


data['diff_ln_S&P500'] = data['ln_S&P500'].diff()
data['diff_ln_GDP'] = data['ln_GDP'].diff()
data['diff_Inflation_Rate'] = data['Inflation Rate'].diff()
data['diff_Interest_Rate'] = data['Interest Rate'].diff()


# In[16]:


result = coint_johansen(data[['ln_S&P500', 'ln_GDP', 'Inflation Rate', 'Interest Rate']], det_order=0, k_ar_diff=1)


# In[17]:


cv = result.evec[:, 0]
cv = cv / cv[0]


# In[18]:


data['ec_term'] = cv[0] * data['ln_S&P500'] + cv[1] * data['ln_GDP'] + cv[2] * data['Inflation Rate'] + cv[3] * data['Interest Rate']
data['diff_ec_term'] = data['ec_term'].diff()


# In[19]:


model = sm.OLS(data['diff_ln_S&P500'].iloc[1:], sm.add_constant(data[['diff_ln_GDP', 'diff_Inflation_Rate', 'diff_Interest_Rate', 'diff_ec_term']].iloc[1:]))
results = model.fit()
print(results.summary())


# In[20]:


# Heteroskedasticity test (White test)
white_test = sm.stats.diagnostic.het_white(results.resid, results.model.exog)
print("White test p-value:", white_test[-1])

# Autocorrelation test (Breusch-Godfrey test)
bg_test = sm.stats.diagnostic.acorr_breusch_godfrey(results)
print("Breusch-Godfrey test p-value:", bg_test[-1])

# Normality test (Jarque-Bera test)
jb_test = sm.stats.stattools.jarque_bera(results.resid)
print("Jarque-Bera test p-value:", jb_test[-1])


# In[21]:


model = sm.OLS(data['diff_ln_S&P500'].iloc[1:], sm.add_constant(data[['diff_ln_GDP', 'diff_Inflation_Rate', 'diff_Interest_Rate', 'diff_ec_term']].iloc[1:]))
results = model.fit(cov_type='HC0')
print(results.summary())


# In[ ]:




