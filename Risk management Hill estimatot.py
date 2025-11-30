import os
import marshal
import types
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scipy.stats import expon, norm
from scipy.special import gammainc, gamma
os.getcwd()


#collecting the insurance data (car insurance claims)and the fire insurance randomly generated data
np.random.seed(42)
car_insurance_claims = np.random.exponential(scale=5000, size=1000)
fire_insurance_claims = np.random.exponential(scale=20000, size=1000)
#adding both the data at this point
total_claims = car_insurance_claims + fire_insurance_claims

# Define n as the length of total_claims
#checking the qq plot for the total claims of the data

n = len(total_claims)
total_claims_sorted = np.sort(total_claims)[::-1] # here putting it in the descending order
exponential_theortical_quantiles = [
    expon.ppf((n - k + 1) / (n + 1), scale=np.mean(total_claims)) for k in range(1, n + 1)
]
plt.figure(figsize=(8,6))
plt.scatter(exponential_theortical_quantiles, total_claims_sorted)
plt.plot([0, max(total_claims_sorted)], [0, max(total_claims_sorted)], color='red', linestyle='--')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Empirical Quantiles')
plt.title('QQ Plot - Total Insurance Claims vs Exponential Distribution')
plt.grid()
plt.show()
# Hill Estimator Function using the bootstrap method
def hill_estimator(data):
    sorted_data = np.sort(data)[::-1]
    hill_estimator = np.zeros(n-1)
    for k in range(2, n+1):
        hill_estimator[k-2] = k /(np.sum(np.log(sorted_data[:k-1])- np.log(sorted_data[k-1])))

    return hill_estimator
hill_estimates = hill_estimator(total_claims)
# Plotting the Hill estimates

plt.figure(figsize=(10,6))
plt.plot(range(2, n), hill_estimates, marker='o')
plt.xlabel('k (Number of Top Order Statistics)')
plt.ylabel('Hill Estimator')
plt.title('Hill Estimator for Tail Index Estimation')
plt.grid()
plt.show()

#now i would like to calculate the VaR and the ES using the hill estimator
def calculate_VaR(data, hill_estimator, alpha = 0.99):
    sorted_data = np.sort(data)[::-1]
    k =np.arange(2, n+1)
    var_estimates = (n/k * (1-alpha))**(-1/hill_estimator)*sorted_data[k-2]
    return var_estimates

def calculate_ES(data, hill_estimator, alpha = 0.99):




