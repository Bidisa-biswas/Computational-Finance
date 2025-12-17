import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2


file_path = 'UK_election dataset' # Please add the UK_data here in csv format
df = pd.read_csv(file_path)#Will read the data from the file path

df = pd.DataFrame(file_path)# converting the file path data into the dataframe structure
df =df.dropna()# dropping the missing values from the data

#Prepare the data for the regression
y = df['y']# this is the dependent variable
dependent_vars = df[['x1','x2','x3','x4']]# these are the independent variables
x = sm.add_constant(dependent_vars)# adding the constant to the independent variables

#Using OLS regression to fit the model
model = sm.OLS(y,x).fit()#you are fitting the model with the dependent and the independent variables
print("OLS Regression Results:\n", model_ols.summary())#printing the summary of the model

#Testing thehypothesis given as b1 = b2 = 0
#H0: b1 = b2 = 0
#H1: b1 != 0 or b2 != 0
#LM-test to check for the restricted and the unrestricted model
x_restricted =  sm.add_constant(df[['x3','x4']])# this is the restricted model
model_restricted = sm.OLS(y,x_restricted).fit()#fitting the restricted model
restricted_residuals = model_restricted.resid#calculating the residuals of the restricted model

#calculating the residuals of the unrestricted model
unrestricted_residuals = model.resid#calculating the residuals of the unrestricted model

#Calculated the LM test statistic
n = len(df)
lm_test_stat = n*(np.sum(restricted_residuals**2)-np.sum(unrestricted_residuals**2))/(np.sum(unrestricted_residuals**2))
p_value = 1-chi2.cdf(lm_test_stat,2)#calculating the p value
print("LM Test Statistic:", lm_test_stat)
print("P-value:", p_value)

#F-Test
rss_r = model_restricted.ssr
rss_ur = model_ols.ssr
F = ((rss_r - rss_ur) / 2) / (rss_ur / (n - x.shape[1]))
print(f"\nF-Test Statistic: {F}")

#calculating the 2SLS regression
stage1 = sm.OLS(df['x4'], sm.add_constant(df['constituency'])).fit()
df['x4_hat'] = stage1.fittedvalues
X_iv = sm.add_constant(df[['x1', 'x2', 'x3', 'x4_hat']])
model_2sls = sm.OLS(y, X_iv).fit()
print("\n2SLS Regression Results:\n", model_2sls.summary())

# Compare OLS and 2SLS Beta_4
print(f"\nOLS Beta_4: {model_ols.params['x4']}, 2SLS Beta_4: {model_2sls.params['x4_hat']}")




