import statsmodels.tsa.api as smt
from .. import Utils as utils


import pmdarima

def plot_ARIMA_forecast(df_train, df_test, col="Close", summary=True):
    arima_model = pmdarima.auto_arima(df_train[col], exogenous=None,
                                 seasonal=True, stationary=False,d=2,
                                 m=4, information_criterion='aic',
                                 error_action='ignore',trace=True)
    if summary:
        print("best model --> (p, d, q):",  arima_model.order, " and  (P, D, Q, s):", arima_model.seasonal_order)
    fit_sarimax(df_train, df_test, order=arima_model.order, seasonal_order=arima_model.seasonal_order, col=col, summary=summary, figsize=(12,8))


def fit_sarimax(df_train, df_test, order,
                seasonal_order, figsize, col="Close", summary=True, exog_train=None,
                exog_test=None):
    ## train
    model = smt.SARIMAX(df_train[col], order=order,
                        seasonal_order=seasonal_order,
                        exog=exog_train, enforce_stationarity=True,
                        enforce_invertibility=True).fit()
    df_train["model"] = model.fittedvalues

    ## test
    df_test["forecast"] = model.predict(start=len(df_train),
                                         end=len(df_train) + len(df_test) - 1,
                                         exog=exog_test)

    ## evaluate
    df = df_train.append(df_test)
    df["ts"] = df["Close"]
    utils.plot_forecast(df=df, title="Auto Arima Forecast", summary=summary)
