from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .. import Utils as utils

def plot_hw_model(df_train, df_test, periods, col="Close", summary=True, seasonal_type='add', damped=True):
    hw_model = ExponentialSmoothing(df_train[col], trend="add", seasonal=seasonal_type, seasonal_periods=periods, damped=damped).fit()
    hw_predictions = hw_model.predict(start=0, end=len(df_train)+len(df_test))
    df_train["model"] = hw_predictions[:len(df_train)]
    df_test["forecast"] = hw_predictions[len(df_train):len(df_train) + len(df_test)]
    df = df_train.append(df_test)
    df["ts"] = df[col]
    utils.plot_forecast(df=df, title="Holt-Winters Forecast", summary=summary)
    # df = df_train.append(df_test)
    # df["ts"] = df["Close"]
    # print(df)
    # utils.plot_forecast(df=df, title="Holt-Winters Forecast", summary=True)
    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(FuncFormatter(format_date))
    # fig.autofmt_xdate()
    # ax.plot(hw_predictions.index, hw_predictions)
    # ax.plot(df.index, df["Close"])
    # return ExponentialSmoothing(train_data, trend='add', seasonal=type, damped=damped, seasonal_periods=periods)
