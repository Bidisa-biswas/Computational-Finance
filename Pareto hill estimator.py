# Generate 2000 Pareto-distributed data points
# Note: numpy's pareto uses shape parameter a (alpha) = k
# For general Pareto distribution, we need to use scale parameter
#     sorted_data = np.sort(data)[::-1]


import os
import marshal
import types
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scipy.stats import expon, norm
from scipy.special import gammainc, gamma
os.getcwd()

#generating the 2000 pareto distribution data points
np.random.seed(42)
pareto_data =(np.random.pareto(a=2.5, size=2000) + 1) * 1000# scale parameter of 1000
# Define n as the length of pareto_data
n = len(pareto_data)
#checking the qq plot of the pareto data that has the heavy tails with the theoritical quantiles
pareto_data_sorted = np.sort(pareto_data)[::-1] # here putting it in the descending order
data = pareto_data
k = np.arange(1, n+1)
exponential_quantiles = expon(scale= np.mean(data)).ppf((n-k+1)/(n+1))
plt.figure(figsize=(8,6))
plt.scatter( pareto_data_sorted, exponential_quantiles, color='blue', alpha=0.6)
plt.plot([0, max(pareto_data_sorted)], [0, max(pareto_data_sorted)], color='red', linestyle='--')
plt.xlabel('Emopirical Quantiles')
plt.ylabel('Theoritical Quantiles')
plt.show()

#Estimating the Hill estimator
hill_estimates = np.zeros(n-1)
for k in range(2, n+1):
    hill_estimates[k-2] = k /(np.sum(np.log(pareto_data_sorted[:k-1])- np.log(pareto_data_sorted[k-1])))
# Plotting the Hill estimates
plt.figure(figsize=(10,6))
plt.plot(range(2, n), hill_estimates, marker='o')
plt.xlabel('k (Number of Top Order Statistics)')
plt.ylabel('Hill Estimator')
plt.title('Hill Estimator for Tail Index Estimation')
plt.grid()
plt.show()
# The Hill estimator plot shows the estimated tail index for different values of k.
# From the plot, we can observe that the Hill estimator stabilizes around a certain value as k increases.
# This stabilized value gives us an estimate of the tail index of the Pareto distribution.
# In this case, since we generated the data from a Pareto distribution with shape parameter 2.5,
# we expect the Hill estimator to converge to a value close to 2.5 for sufficiently large k.

#estimating the hill estimator usimng the bootstrap method
#drawing bootstrap sample from the pareto distribution
def hill_estimator_bootstrap(data, num_boostrap= 1000 ):
    n = len(data)
    hill_estimates_bootstrap = np.zeros((num_boostrap, n-1))
    for b in range(num_boostrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        sorted_sample = np.sort(bootstrap_sample)[::-1]
        for k in range(2, n+1):
            hill_estimates_bootstrap[b, k-2] = k /(np.sum(np.log(sorted_sample[:k-1])- np.log(sorted_sample[k-1])))
    return hill_estimates_bootstrap

hill_estimates_bootstrap = hill_estimator_bootstrap(pareto_data)
#calculate the MSE, bias and variance from the bootstrap samples
mse_bootstrap = np.mean((hill_estimates_bootstrap - hill_estimates)**2, axis=0)
bias_bootstrap = (np.mean(hill_estimates_bootstrap, axis=0) - hill_estimates)
variance_bootstrap = np.var(hill_estimates_bootstrap, axis=0)
# Plotting the MSE, Bias, and Variance from Bootstrap

plt.figure(figsize=(12,8))

plt.plot(range(2, n), mse_bootstrap, label='MSE', color='blue')
plt.plot(range(2, n), bias_bootstrap, label='Bias', color='orange')
plt.plot(range(2, n), variance_bootstrap, label='Variance', color='green')
plt.xlabel('k (Number of Top Order Statistics)')
plt.ylabel('Value')
plt.title('Bootstrap Estimates of MSE, Bias, and Variance of Hill Estimator')
plt.legend()
plt.grid()
plt.show()
# The plots of MSE, Bias, and Variance from the bootstrap samples provide insights into the performance of the Hill estimator.
# From the plots, we can observe the following:
# 1. MSE: The Mean Squared Error (MSE) tends to decrease as k increases, indicating that the estimator becomes more accurate with larger k.
# 2. Bias: The bias plot shows how the average estimate from the bootstrap samples deviates from the original Hill estimator.
#    Ideally, we want the bias to be close to zero, indicating that the estimator is unbiased.
# 3. Variance: The variance plot indicates the variability of the Hill estimator across different bootstrap samples.
#    A lower variance suggests that the estimator is stable and reliable.

# Overall, these plots help us understand the reliability and accuracy of the Hill estimator for tail index estimation in heavy-tailed distributions.

#finding the confidence intervals for the hill estimator using the bootstrap method
alpha = 0.05
lower_bound = np.percentile(hill_estimates_bootstrap, 100 * (alpha / 2), axis=0)
upper_bound = np.percentile(hill_estimates_bootstrap, 100 * (1 - alpha / 2), axis=0)
# Plotting the Hill estimates with Confidence Intervals
plt.figure(figsize=(10,6))
plt.plot(range(2, n), hill_estimates, marker='o', label='Hill Estimator', color='blue')
plt.fill_between(range(2, n), lower_bound, upper_bound, color='orange', alpha=0.3, label='95% Confidence Interval')
plt.xlabel('k (Number of Top Order Statistics)')

#finding the survival function using the hill estimator
plt.ylabel('Hill Estimator')
plt.title('Hill Estimator with 95% Confidence Intervals')
plt.legend()
plt.grid()
plt.show()



