import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from scipy import optimize as sco

enddate= pd.to_datetime('today').normalize()
startdate = enddate-dt.timedelta(2520)
startdate

assets = ['KO', 'PFE', 'AMZN','V', 'PEP','DIS','TSLA']
noa = len(assets)
data = yf.download(assets, start=startdate, end=enddate)['Adj Close']
prices = yf.download(assets, period = '2y')['Adj Close'].pct_change()
returns = prices
rets = np.log(data / data.shift(1))
rets.hist(bins=40, figsize=(10, 8));

"""Class creation of the calculations for mean, variance, and the correlation matrix"""

# Credit: https://github.com/CCNY-Analytics-and-Quant/Efficient-Frontier-Project-Qetsiyah-Osamwonyi/blob/main/EF_code.ipynb
class Calculations:
    def __init__(self, returns, assets):
        self.returns = returns
        self.assets = assets

    def computation(self):
        self.tbl = pd.DataFrame(index = self.assets)
        self.tbl['Mean'] = returns.mean()
        self.tbl['Variance'] = returns.var()
        return(self.tbl.T)

    def correlation(self):
        self.tbl = pd.DataFrame(index=self.assets)
        self.corr = returns.corr()
        return(self.corr)

corr_mean_var_calc = Calculations(returns = returns, assets=assets)
display(corr_mean_var_calc.computation().T, corr_mean_var_calc.correlation())

plt.figure(figsize=(10, 6))
sns.heatmap(corr_mean_var_calc.correlation(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

weights = np.random.random(noa)
weights /= np.sum(weights)
print(weights)

def port_ret(weights):
 return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
 return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

"""Monte Carlo"""

prets = []
pvols = []
for p in range (50000):
   weights = np.random.random(noa)
   weights /= np.sum(weights)
   prets.append(port_ret(weights))
   pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols,
            marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio');

def min_func_sharpe(weights):
    return -port_ret(weights) / port_vol(weights)

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
eweights = np.array(noa * [1. / noa,])

opts = sco.minimize(min_func_sharpe, eweights,
                    method='SLSQP', bounds=bnds,
                    constraints=cons)

opts['x'].round(3)
port_ret(opts['x']).round(3)
port_vol(opts['x']).round(3)
port_ret(opts['x']) / port_vol(opts['x'])

optv = sco.minimize(port_vol, eweights,
                    method='SLSQP', bounds=bnds,
                    constraints=cons)

print(optv['x'].round(3),
port_vol(optv['x']).round(3),
port_ret(optv['x']).round(3),
port_ret(optv['x']) / port_vol(optv['x']))

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x)-tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x)- 1})

bnds = tuple((0, 1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP',
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols,
            marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']),
         'y*', markersize=15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']),
         'r*', markersize=15.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')