import streamlit as st
from datetime import date, timedelta

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd, numpy as np
import plotly.express as px

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

# Change the following line to use st.selectbox for choosing the number of years to predict
n_years_future = st.sidebar.selectbox('Select number of years to predict into the future:', list(range(1, 11)))
period_future = n_years_future * 365

@st.cache_data
def load_data(ticker):
    start_date = "2015-01-01"
    today_date = date.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start_date, today_date)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data)

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

data2 = data
data2['% Change']=data['Adj Close']/data['Adj Close'].shift(1)-1
data2.dropna(inplace = True)

annual_return = data2['% Change'].mean()*252*100
st.write('Annual Return is ',annual_return,'%')
stdev = np.std(data2['% Change'])*np.sqrt(252)
st.write('Standard Deviation is ',stdev*100,'%')
st.write('Risk Adj. Return is ',annual_return/(stdev*100))

    

start_prophet = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period_future, include_history=False)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast)

st.write(f'Forecast plot for {n_years_future} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


# Vertical Tabs
tabs = ["Forecast Components", "Comparison Charts","Fundamental Data","Top 10 News","Technical Indicators"]
selected_tab = st.sidebar.radio("Select Tab:", tabs)

if selected_tab == "Forecast Components":
    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

elif selected_tab == "Comparison Charts":
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stocks = st.sidebar.multiselect('Select datasets for Comparison', stocks)

    # Plot forecast for multiple stocks
    def plot_forecast_multiple(stocks):
        fig = go.Figure()

        for stock in stocks:
            data = load_data(stock)

            # Predict forecast with Prophet
            start_prophet = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period_future, include_history=False)
            forecast = m.predict(future)

            # Plot forecasted data
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"{stock} - Forecasted Close"))

        fig.layout.update(title_text='Forecast for Multiple Stocks', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    # Call the function to plot forecasted data for multiple stocks
    plot_forecast_multiple(selected_stocks)

elif selected_tab == "Fundamental Data":
    from alpha_vantage.fundamentaldata import FundamentalData
    key = " Z5ULTIJK2DSHRF1Q"
    fd = FundamentalData(key,output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(selected_stock)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(selected_stock)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(selected_stock)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)

elif selected_tab == "Top 10 News":
    from stocknews import StockNews
    st.header(f'News of {selected_stock}')
    sn = StockNews(selected_stock,save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

elif selected_tab == "Technical Indicators":
    import pandas_ta as ta
    st.subheader('Technical Analysis Dashboard: ')
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    technical_indicator = st.selectbox('Tech Indicator',options=ind_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta,method)(low=data['Low'],close=data['Close'],high=data['High'],open=data['Open'],volume=data['Volume']))
    indicator['Close']=data['Close']
    st.write(indicator) 
    figw_ind_new = px.line(indicator)
    st.plotly_chart(figw_ind_new)
