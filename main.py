import numpy as np
import datetime as dt
import yfinance as yf
import scipy as sc
import pandas as pd
import plotly.graph_objects as go

def get_data(stocks, start, end):
    closes = yf.download(" ".join(stocks), start=start, end=end)['Close']
    returns = closes.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def calc_negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0):
    returns, std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(returns-risk_free_rate)/std

def portfolio_variance(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def portfolio_returns(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def max_sharpe(mean_returns, cov_matrix, risk_free_rate=0, contraint_set=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(contraint_set for _ in range(num_assets))
    result = sc.optimize.minimize(calc_negative_sharpe, num_assets * [1./num_assets], args, method='SLSQP', bounds=bounds, constraints=constraints)
    return -result['fun'], result['x']

def minimize_variance(mean_returns, cov_matrix, risk_free_rate=0, contraint_set=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(contraint_set for _ in range(num_assets))
    result = sc.optimize.minimize(portfolio_variance, num_assets * [1./num_assets], args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result['fun'], result['x']


def efficient_opt(mean_returns, cov_matrix, return_target, constraint_set=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: portfolio_returns(x, mean_returns, cov_matrix) - return_target})
    bounds = tuple(constraint_set for _ in range(num_assets))
    result = sc.optimize.minimize(portfolio_variance, num_assets * [1./num_assets], args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result



def calculatedResults(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0,1)):
    max_sr, max_sharpe_weights = max_sharpe(mean_returns, cov_matrix)
    max_sr_returns, max_sr_std = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
    max_sr_allocation = pd.DataFrame(max_sharpe_weights, index=mean_returns.index, columns=['allocation'])
    max_sr_allocation.allocation = [round(i*100, 2) for i in max_sr_allocation.allocation]
    print(max_sr_allocation.to_string())
    print(max_sr_returns, max_sr_std)

    min_var, min_var_weights = minimize_variance(mean_returns, cov_matrix)
    min_var_returns, min_var_std = portfolio_performance(min_var_weights, mean_returns, cov_matrix)
    min_var_allocation = pd.DataFrame(min_var_weights, index=mean_returns.index, columns=['allocation'])
    min_var_allocation.allocation = [round(i*100, 2) for i in min_var_allocation.allocation]

    efficientList = []
    target_returns = np.linspace(min_var_returns, max_sr_returns, 20)
    for target in target_returns:
        efficientList.append(efficient_opt(mean_returns, cov_matrix, target)['fun'])

    max_sr_returns, max_sr_std = round(max_sr_returns*100, 2), round(max_sr_std*100, 2)
    min_var_returns, min_var_std = round(min_var_returns*100, 2), round(min_var_std*100, 2)


    return max_sr_returns, max_sr_std, max_sr_allocation, min_var_returns, min_var_std, min_var_allocation, efficientList, target_returns

def efficient_frontier_graph(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0, 1)):
    max_sr_returns, max_sr_std, max_sr_allocation, min_var_returns, min_var_std, min_var_allocation, efficientList, target_returns = calculatedResults(mean_returns, cov_matrix)

    MaxSharpeRatio = go.Scatter(
        name="Maximum Sharpe Ratio",
        mode="markers",
        x=[max_sr_std],
        y=[max_sr_returns],
        marker=dict(color="red", size=14, line=dict(width=3, color="black"))
    )

    MinVar = go.Scatter(
        name="Minimum Volatility",
        mode="markers",
        x=[min_var_std],
        y=[min_var_returns],
        marker=dict(color="green", size=14, line=dict(width=3, color="black"))
    )

    EfficientFrontierCurve = go.Scatter(
        name="Efficient Frontier",
        mode="lines",
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target*100, 2) for target in target_returns],
        line=dict(color="black", width=4, dash="dashdot")
    )

    data = [MaxSharpeRatio, MinVar, EfficientFrontierCurve]
    layout = go.Layout(
        title="Portfolio Optimization",
        yaxis=dict(title="Annualized Return (%)"),
        xaxis=dict(title="Annualized Volatility (%)"),
        showlegend=True,
        legend=dict(
            x=0.75, y=0, traceorder='normal', bgcolor="#E2E2E2", bordercolor="black", borderwidth=2
        ),
        width=800,
        height=600
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html( 'output_file_name.html', 
                   auto_open=True )
    return fig.show()

stockList = sorted(["AAPL", "MSFT", "AMZN", "GOOG"])
weights = np.array([1/len(stockList)] * len(stockList))
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

mean_returns, cov_matrix = get_data(stockList, start=startDate, end=endDate)
efficient_frontier_graph(mean_returns, cov_matrix)