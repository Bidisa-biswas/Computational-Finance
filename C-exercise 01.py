# Part a) European call optiuon exercise 01 sheet 001
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


# defining the function for the CRR model with the u and d

def stock_price(S0, r, sigma, T, M, K):
    # the change in the time
    delta_T = T / M
    # the upward value
    theta = np.exp(r * delta_T)
    beta = 0.5 * (1 / theta + np.exp(r + sigma ** 2) * delta_T)
    u = beta + np.sqrt(beta ** 2 - 1)
    # the downward movement
    d = 1 / u
    # finding the equivalent martingale measure (EMMQ)
    q = (np.exp(r * delta_T) - d) / (u - d)

    # Find out the stock price

    S = np.zeros((M + 1, M + 1))  # for calculating the rows and the coloums
    for j in range(M + 1):
        for i in range(j + 1):
            S[i, j] = S0 * (u ** (j - i)) * (d ** i)

    return S


# stock_price(100, 0.03, 0.3, 1, 100, 0)

# b)
# finding the european call option
def CRR_european_call_option(S0, r, sigma, T, M, K):
    delta_T = T / M
    Theta = np.exp(r * delta_T)
    beta = 0.5 * (1 / Theta + np.exp(r + sigma ** 2) * delta_T)
    u = beta + np.sqrt(beta ** 2 - 1)
    d = 1 / u
    q = (np.exp(Theta) - d) / (u - d)

    # finding the stock price for the time period n and with the M steps in the future
    S = np.zeros((M + 1, M + 1))  # for calculating the rows and the coloums
    for j in range(M + 1):
        for i in range(j + 1):
            S[i, j] = S0 * (u ** (j - i)) * (d ** i)

    # finding the payoff stock value

    # creating the matrix for the corresponding stock price
    V = np.empty((M + 1, M + 1))
    # calculating the maximum value from for the non-negative stock value
    V[:, M] = np.maximum(S[:, M] - K, 0)  # this calculates the maximum payoff non negative

    def recursive(n):
        return np.exp(-r * delta_T) * (q * V[1:n + 1, n] + (1 - q) * V[0:n, n])

    for n in range(M, 0, -1):  # here the
        V[0:n, n - 1] = recursive(n)

    # return the initial price
    return V[0, 0]


# c) the black scholes model :

def Black_scholes_model(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t / K) + (r + (np.power(sigma, 2) / 2) * (T - t))) / (sigma * np.sqrt(T - t))
    d2 = d1 - (sigma * np.sqrt(T - t))
    call_option = S_t * ss.norm.cdf(d1) - K * np.exp(-r * (T - t)) * ss.norm.cdf(d2)
    return call_option


# values given=
S0 = 100
r = 0.03
sigma = 0.3
T = 1
M = 100
K = range(70, 200)

# generating the matrices for the CRR and the BS_model as we have different K prices

V_CRR_model = np.empty(len(K), dtype=float)
V_BS_model = np.empty(len(K), dtype=float)

# now the above K (strike price) model is created to find out the values for the different K price
# generating the european call option prices for different K as we have set for different ranges
for i in range(0, len(K)):
    V_CRR_model[i] = CRR_european_call_option(S0, r, sigma, T, M, K[i])
    V_BS_model[i] = Black_scholes_model(S0, r, sigma, T, M, K[i])

# Plotting the graphs and the error value:

plt.clf()
plt.plot(V_CRR_model)
plt.plot(V_BS_model)
plt.plot(K, V_BS_model - V_CRR_model, 'g')
plt.xlabel('K = Strike price')
plt.ylabel('Difference')
plt.title('European call option pricing')
plt.show