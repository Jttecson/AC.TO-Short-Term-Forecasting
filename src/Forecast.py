import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pylab import rcParams
import src.Models.Regression as reg_models
import src.Models.ETS as ets_models
import src.Models.ARIMA as arima_models
import src.Models.Random_Walk as rw_models
import src.Models.Historical_Averages as ha_models

def plot_ols_forecast():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    ols_df = df[["Date","Monday","Tuesday","Wednesday","Thursday","Close"]]
    ols_df["Indices"] = ols_df.index
    df_train, df_test = split_data_test_train(ols_df)
    reg_models.plot_ols_forecast(df_train=df_train, df_test=df_test, col="Close", summary=True)

def plot_holt_winters_forecast():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_train, df_test = split_data_test_train(df)
    ets_models.plot_hw_model(df_train, df_test, 4, col="Close", summary=True, seasonal_type="add", damped=False)

def plot_ARIMA_forecast():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_train, df_test = split_data_test_train(df)
    sm.graphics.tsa.plot_acf(df_train["Close"],lags=40)
    ts = df_train["Close"].diff()
    sm.graphics.tsa.plot_acf(ts[1:], lags=40)
    sm.graphics.tsa.plot_pacf(ts[1:], lags=40)
    ts = ts.diff()
    sm.graphics.tsa.plot_acf(ts[2:], lags=40)
    sm.graphics.tsa.plot_pacf(ts[2:], lags=40)
    arima_models.plot_ARIMA_forecast(df_train=df_train, df_test=df_test, col="Close")

def historical_median():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_train, df_test = split_data_test_train(df)
    ha_models.plot_historical_median(df_train=df_train, df_test=df_test)

def historical_mean():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df_train, df_test = split_data_test_train(df)
    ha_models.plot_historical_mean(df_train=df_train, df_test=df_test)

def random_walk():
    df = pd.read_csv("./src/AC_TO Pre-Processed - AC.TO.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    train, test = split_data_test_train(df)
    rw_models.plot_random_walk(train, test)

def split_data_test_train(df, test_percent = 0.22):
    train_len = int(len(df) * (1-test_percent))
    train_df = df.head(train_len)
    test_df = df.tail(len(df) - train_len)
    return train_df, test_df

def format_date(index,pos):
    df = pd.read_csv("AC_TO Pre-Processed - Valid Dates.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    fmt = "%Y-%m-%d"
    index = np.clip(int(index + 0.5), 0, len(df['Date'])-1)
    return df['Date'][index].strftime(fmt)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    plt.style.use('fivethirtyeight')
    rcParams['figure.figsize'] = 10, 6
    plot_ols_forecast()
    plot_holt_winters_forecast()
    plot_ARIMA_forecast()
    historical_median()
    historical_mean()
    random_walk()
    plt.show()


