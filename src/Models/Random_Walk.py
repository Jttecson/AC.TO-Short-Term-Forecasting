from .. import Utils as utils
import numpy as np


def plot_random_walk(df_train, df_test, col="Close", summary="True", drift="True"):
    b = df_train[col].iloc[0]
    m = (df_train[col].iloc[-1] - df_train[col].iloc[0]) / len(df_train)
    df_train["model"] = df_train.index * m + b
    df_test["forecast"] = (df_test.index) * m + b
    df = df_train.append(df_test)
    df["ts"] = df["Close"]
    utils.plot_forecast(df=df, title="Random Walk Forecast", summary=summary)
