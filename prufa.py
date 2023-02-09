import streamlit as st

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


START = "2015-06-01"
INTERVAL = "1wk"
stock_data = yf.download(
    tickers = TICKERS,
    start = START,
    interval = INTERVAL
).dropna()

#st.write(stock_data)


TICKERS_selection = []


#Dsiplay checkboxes
col1, col2, col3 = st.columns(3)

for i,tick in enumerate(TICKERS):
    if i<5:
        if col1.checkbox(tick):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    elif 5<=i<10:
        if col2.checkbox(tick):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    else:
        if col3.checkbox(tick):
            if tick not in TICKERS_selection:
                TICKERS_selection += [tick]
            else:
                TICKERS_selection.remove(tick)
    




# Data from exercise sheet 7
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

    mus = np.linspace(np.min(mu) - np.ptp(mu), np.max(mu) + np.ptp(mu))
    sigmas = np.sqrt((c*mus**2 - 2*b*mus + a) / (a*c - b**2)).squeeze()

    sigma_p = np.linspace(0, 1)
    mu_e = mu - RFR
    mu_p = (sigma_p * np.sqrt(mu_e.T @ Sigma_inv @ mu_e) + RFR).squeeze()

    mu_e_tan = (mu_e.T @ Sigma_inv @ mu_e) / (np.ones_like(mu_e).T @ Sigma_inv @ mu_e)
    sigma_tan = np.sqrt(mu_e.T @ Sigma_inv @ mu_e) / (np.ones_like(mu_e).T @ Sigma_inv @ mu_e)
    slope = mu_e_tan / sigma_tan

    #plt.figure(figsize=(9,6))
    #plt.plot(sigmas, mus, label="Efficient Frontier 1")
    #plt.plot(sigmas, mus, label="Efficient Frontier 2")
    #plt.plot(sigma_p, (sigma_p*slope+RFR).squeeze(), label="Capital Market Line 1")
    #plt.plot(sigma_p, (sigma_p*slope_b+RFR).squeeze(), label="Capital Market Line 2")
    #plt.plot(np.diag(Sigma_b)**0.5, mu_b, "x", mew=6, label="Assets")
    #plt.plot(sigma_tan, mu_e_tan + RFR, "x", mew=6, label="Tangent Portfolio 1")
    #plt.plot(sigma_tan_b, mu_e_tan_b + RFR, "x", mew=6, label="Tangent Portfolio 2")
    #plt.legend(loc="lower right")
    #plt.xlim([0.0, 0.4])
    #plt.ylim([0.0, 0.6])

    #plt.figure(figsize=(9*2,6*2))
    #plt.plot(sigmas, mus, label="Efficient Frontier")
    #plt.plot(sigma_p, (sigma_p*slope+RFR).squeeze(), label="Capital Market Line")
    #plt.plot(np.diag(Sigma)**0.5, mu, "x", mew=6, label="Assets")
    #plt.plot(sigma_tan, mu_e_tan + RFR, "x", mew=6, label="Tangent Portfolio")
    #plt.xlim([0.0, 0.5])
    #plt.ylim([0.0, 0.4])
    #plt.legend(loc="lower right")
    #None

    fig = go.Figure(data=[go.Scatter(x=sigmas, y=mus, line=dict(color='red', width=2)),
                        go.Scatter(x=sigma_p, y=(sigma_p*slope+RFR).squeeze(), line=dict(color='blue', width=2)),
                        go.Scatter(x=np.diag(Sigma)**0.5, y=mu.squeeze(),mode='markers', marker=dict(color='green', size=7))])

    fig.update_layout(title='Efficient Frontier',
                    xaxis_title='X',
                    yaxis_title='Y',
                    plot_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(text=TICKERS_selection, hoverinfo='text')
    st.plotly_chart(fig)

#st.write((mu.squeeze()).shape)
#st.write(len(TICKERS_selection))












#x = np.linspace(0, 10, 100)
#y = np.sin(x)

#line_style = False

#if st.checkbox('Show Dotted Line', value=line_style):
#    line_style = not line_style

#if line_style:
#    fig = go.Figure(data=[go.Scatter(x=x, y=y, line=dict(color='blue', width=2, dash='dot'))])
#else:
#    fig = go.Figure(data=[go.Scatter(x=x, y=y, line=dict(color='red', width=2))])

#fig.update_layout(title='Interactive Line Chart',
#                  xaxis_title='X',
#                  yaxis_title='Y')
#st.plotly_chart(fig)


st.write("Thank you for trying Streamlit!")


