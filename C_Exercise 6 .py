import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stat


# defining the function for both the american and the european option
def CRR_AmEuPut(S_0, r , sigma, T, M, K, Eu):
  #M = M.stop # for change the range to integer
  Delta_t = T/M
  theta = np.exp(r * Delta_t)# here this code is assigned to defing the rate of the riskless assets in the continuous time frame
  beta = 0.5 * (1/theta) + np.exp((r + sigma**2)* Delta_t) #here defining the beta for the calculation of u = upward and d = downward movement from the lecture note - 1.6
  u = beta + np.sqrt((beta ** 2) - 1)  # lecture note - 1.4
  d = 1/u # lecture note - 1.5
  q = (theta - d) / (u - d) # the probability calculation given in Lecture note - 1.7

#the matrix for the stock prices
  S_1 = np.empty((M+1,M+1))

  # using the for loop to interate the function
  for i in range(M+1):
    for j in range (M+1):
      S_1[j,i] = S_0 * np.power(u,i) * np.power(d, j)

  # matrix for the value of the option
  V_n = np.empty((M+1, M+1))

  # Calculating the payoff value of the put option with respect tor the formula maximum(K-S)in all the iteration until step M and checking for the maximum value, otherwise zero.
  V_n[:M] = np.maximum(K - S_1[:M], 0)

  # In order to calculate the V_0 or the initial value of the option value, we use the recursive method for the backward calculation. According to the question, if Eu is 1 then it will return the european option, and if Eu=o then it will return the american option
  if Eu == 1:
    def Recursive(i):
      return 1/theta * (q * (V_n[1:i],i)+(1-q)*(V_n[0:i],i)) # referrening the lecture note formula : 1.9 for the european put option
  else:
    def Recursive(i):
      return np.maximum(1/theta * (q * (V_n[1:i],i)+(1-q)*(V_n[0:i],i)), K- S_1[0:1,i]) #this is uses the maximum of the expected value with respect to the iterated value i or the higher payoffs

    #finding the fair price value backward from the step M to 0
    for i in range (M,0,-1):
      V_n[0:i,i] = Recursive(i)

 #b : Black scholes model:
def BlackscholesPut(t, S_t, r, sigma, T, K):
  d1 = (math.log(S_t/K) + (r + np.power(sigma, 2)/2) * (T - t))/(sigma* math.sqrt(T - t))
  d2 = d1 - (sigma* np.sqrt(T - t))
  put = K* np.exp(-r*(T-t))*stat.norm.cdf(-d2) - S_t* stat.norm.cdf(-d1)
  return put


#matrix formation for the calculation for the CRR and the BS model
V_CRR = np.empty(501)


#Assigning the value for the algorithm test for the european option with Eu = 1 given in the question
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range( 10 , 501 )
K = 120
Eu = 1

#according to the question -- the dependency on the number of steps M.
for n in M:
  V_CRR[n] = CRR_AmEuPut (S_0, r ,sigma , T, M, K, Eu)


# Assiging the same parameter to check and plot for the
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 120
Eu = 0

V_BS = BlackscholesPut(0, S_0, r, sigma, T, K, Eu)

#plotting both CRR and BS model:

plt.clf()
plt.plot(np.arange(10, 501), V_CRR[10:], 'y', label = 'Line 1')
plt.plot(np.arange(10, 501), V_BS * np.ones(491), label = 'Line 2')
plt.xlabel('Series of M')
plt.ylabel('Put option price')
plt.legend()
plt.show


#computing the initial price
CRR_AmEuPut(100 ,0.05, 0.3, 1, 200, 120, 0)