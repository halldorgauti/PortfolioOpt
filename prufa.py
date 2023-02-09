import streamlit as st
import datetime

def objective_function(x, Sigma):
    return x.T @ Sigma @ x

def efficient_frontier_optimizer(mu, Sigma, constraints = None, bounds = None):
    target_returns = np.linspace(np.min(mu), np.max(mu),400)
    result_success = np.empty_like(target_returns)
    result_returns = np.empty_like(target_returns)
    result_vars = np.empty_like(target_returns)
    #result_w = np.empty_like(target_returns) #reshape this into correct size (array with number of assets and the lenght of target returns)

    for (i, target) in enumerate(target_returns):
        if constraints is None:
            _constraints = []
        else:
            _constraints = constraints.copy()
        C_DESIRED_RETURN = LinearConstraint(A = mu.T - RFR, lb = target, ub = target) # mu_e' w = mu_e_p
        _constraints.append(C_DESIRED_RETURN)

        t = minimize(
            fun = objective_function,
            x0 = np.zeros_like(mu),
            args = (Sigma, ),
            #method = "SLSQP",
            constraints=_constraints,
            bounds=bounds
        )

        result_success[i] = t.success
        result_returns[i] = target
        result_vars[i] = t.fun
        #result_w[i,] = t.x # make this correct
    return result_success.astype(bool), result_returns, result_vars#, result_w


st.title("Efficient frontier calculator")


#if st.checkbox("Show/Hide"):
#    st.text("This text is shown/hidden based on the checkbox status.")

#if st.checkbox("Show"):
#    st.text("This text /hidden based on the checkbox status.")

import numpy as np

import plotly.graph_objs as go

import yfinance as yf
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import gmean

RFR = 0.02

TICKERS = ["ICEAIR.IC", "ARION.IC", "BRIM.IC", "EIK.IC","EIM.IC","FESTI.IC","HAGA.IC",
           "ICESEA.IC","KVIKA.IC","ORIGO.IC","SYN.IC"]



#st.write(stock_data)


TICKERS_selection = []



#Dsiplay checkboxes
col1, col2, col3 = st.columns(3)

for i,tick in enumerate(TICKERS):
    if i<5:
        if col1.checkbox(tick[:-3], value = True):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    elif 5<=i<10:
        if col2.checkbox(tick[:-3]):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    else:
        if col3.checkbox(tick[:-3]):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    



shortallowed = False
if st.checkbox("Shorting Allowed", value = True):
    shortallowed = not shortallowed


slider = st.slider("Data collected from: ", min_value=2018, max_value=2022, value=2018, step=1)

datechosen = datetime.datetime(year=slider, month=1, day=1)
selected_date_str = datechosen.strftime("%Y-%m-%d")
st.write("Selected year: ", selected_date_str)


# Data from exercise sheet 7


START = "2015-06-01"
INTERVAL = "1wk"
stock_data = yf.download(
    tickers = TICKERS,
    start = selected_date_str,
    interval = INTERVAL
).dropna()

if len(TICKERS_selection) > 1:

    weekly_returns = stock_data["Adj Close"][TICKERS_selection].pct_change().dropna()
    mean_annual_returns = weekly_returns.apply(lambda x: gmean(x + 1)**52 - 1)
    weekly_cov_matrix = weekly_returns.cov()
    annual_cov_matrix = weekly_cov_matrix * 52


    mu = mean_annual_returns.values.reshape(-1,1)
    Sigma = annual_cov_matrix.values
    Sigma_inv = np.linalg.inv(Sigma)
    a = mu.T @ Sigma_inv @ mu
    b = mu.T @ Sigma_inv @ np.ones_like(mu)
    c = np.ones_like(mu).T @ Sigma_inv @ np.ones_like(mu)
    mu_gmv = b/c
    var_gmv = 1/c

    mus = np.linspace(np.min(mu) - np.ptp(mu)*2, np.max(mu) + np.ptp(mu)*2,1000)
    sigmas = np.sqrt((c*mus**2 - 2*b*mus + a) / (a*c - b**2)).squeeze()

    sigma_p = np.linspace(0, 1)
    mu_e = mu - RFR
    mu_p = (sigma_p * np.sqrt(mu_e.T @ Sigma_inv @ mu_e) + RFR).squeeze()

    mu_e_tan = (mu_e.T @ Sigma_inv @ mu_e) / (np.ones_like(mu_e).T @ Sigma_inv @ mu_e)
    sigma_tan = np.sqrt(mu_e.T @ Sigma_inv @ mu_e) / (np.ones_like(mu_e).T @ Sigma_inv @ mu_e)
    slope = mu_e_tan / sigma_tan

 
    if shortallowed:
        fig = go.Figure(data=[go.Scatter(x=sigmas, y=mus, name = "Efficient frontier", line=dict(color='red', width=2)),
                        go.Scatter(x=sigma_p, y=(sigma_p*slope+RFR).squeeze(), name = "CAPM", line=dict(color='blue', width=2)),
                        go.Scatter(x=np.diag(Sigma)**0.5, y=mu.squeeze(), text = TICKERS_selection, hoverinfo = 'text',mode='markers', name = "Assets", marker=dict(color='green', size=7)),
                        go.Scatter(x=sigma_tan[0], y=mu_e_tan[0]+RFR, mode='markers', name = "Tangent Portfolio",marker=dict(color='orange', size=10))])
    else:
        success_a, returns_a, vars_a = efficient_frontier_optimizer(
        mu, Sigma,
        constraints = [LinearConstraint(A = np.ones_like(mu).T, lb = 1, ub = 1)],
        bounds = Bounds(lb=0, ub=np.inf) # w >= 0
        )

        fig = go.Figure(data=[go.Scatter(x=sigmas, y=mus, name = "Efficient frontier", line=dict(color='red', width=2)),
                        go.Scatter(x=sigma_p, y=(sigma_p*slope+RFR).squeeze(), name = "CAPM", line=dict(color='blue', width=2)),
                        go.Scatter(x=np.diag(Sigma)**0.5, y=mu.squeeze(), text = TICKERS_selection, hoverinfo = 'text',mode='markers', name = "Assets", marker=dict(color='green', size=7)),
                        go.Scatter(x=sigma_tan[0], y=mu_e_tan[0]+RFR, mode='markers', name = "Tangent Portfolio",marker=dict(color='orange', size=10)),
                        go.Scatter(x=vars_a[success_a]**0.5, y=returns_a[success_a] + RFR, name = "Efficient frontier", line=dict(dash = 'dash',color='purple', width=2))])


    fig.update_layout(title='Efficient Frontier',
                    xaxis_title='X',
                    yaxis_title='Y',
                    plot_bgcolor='rgba(0,0,0,0)')
    #fig.update_traces(selector=dict(type='scatter', mode='markers'),text=TICKERS_selection, hoverinfo='text')
    fig.update_layout(xaxis=dict(range=[0, max(0.8,mu_e_tan[0,0]*1.3)]),
                  yaxis=dict(range=[-0.6, max(1,mu_e_tan[0,0]*1.3)]))
    st.plotly_chart(fig,use_container_width=True)


st.write(shortallowed)












