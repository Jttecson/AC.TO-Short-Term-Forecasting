import statistics
from .. import Utils as utils

def plot_historical_mean(df_train, df_test, col="Close", summary="True"):
    mean = statistics.mean(df_train[col])
    df_train["model"] = [mean] * len(df_train)
    df_test["forecast"] = [mean] * len(df_test)
    df = df_train.append(df_test)
    df["ts"] = df[col]
    utils.plot_forecast(df=df, title="Historical Mean Forecast", summary=summary)


def plot_historical_median(df_train, df_test, col="Close", summary="True"):
    med = statistics.median(df_train[col])
    df_train["model"] = [med] * len(df_train)
    df_test["forecast"] = [med] * len(df_test)
    df = df_train.append(df_test)
    df["ts"] = df[col]
    utils.plot_forecast(df=df, title="Historical Median Forecast", summary=summary)