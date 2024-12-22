import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp

#fetch the historical data for the stocks
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']# here the equations that you can see is the tickers and the start and the end date will be downloaded form the finance

def portfolio_optimization(returns, number_of_portfolios):
    #calculate the mean daily return and the covariance of the daily return
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    results =np.zeros(4,number_of_portfolios)
    for i in range(number_of_portfolios):
        weights = np.random.random(number_of_assets)
        weights /= np.sum(weights)
        weights = cp.Variable(int(number_of_assets))
        portfolio_return = np.sum(mean_daily_returns*weights *252)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights))*252)# this is the formula for the standard deviation with respect to the value of the portfolio daily return
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = results[0,i]/results[1,i]
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
            data_frame_results = pd.Dataframe(results.T, coloums =['return','stdev', 'sharpe','tickers'], index = range(number_portfolios))
            #constraints of the short selling
            constraints = [cp.sum(weights) == 1, weights >= 0]# you cannot do the short sale in this scenario
            if targeted_return is not None:
                constraints.append(portfolio_return >= targeted_return)
                objective = cp.Maximize(portfolio_return)
                problem = cp.Problem(objective,constraints)
                problem.solve()
            return weights.value, portfolio_return.value,portfolio_std_dev.value, (portfolio_return/portfolio_std_dev).value
#define the stock and the data
tickers = ['AAPL','MSFT','AMZN','GOOGL','FB']
start_date = '2015-01-01'
end_date = '2020-01-01'
number_of_portfolios = 10000
number_of_assets = len(tickers)
targeted_return = 0.2
#fetch the data
stock_data = get_stock_data(tickers,start_date,end_date)#in this scenatio you have all the parameters that you need to get the data

#calculate the daily return
stock_return = stock_data.pct_change()
stock_return = stock_return.dropna()

#run the portfolio optimization
weights, ret, std, sharpe = portfolio_optimization(stock_return, number_of_portfolios)
print('weights', weights)
print('return', ret)
print('std', std)
print('sharpe', sharpe)

#plot the result
plt.figure(10, 6)
plt.bar(range(len(tickers)),weights)
plt.xticks(range(len(tickers)),tickers)
plt.title('Portfolio allocation')
plt.show()# this is the code that you can use to get the portfolio optimization






