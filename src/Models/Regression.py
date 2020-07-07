from .. import Utils as utils
import statsmodels.api as sm

def plot_ols_forecast(df_train, df_test, col="Close", summary="True"):
    dates = df_train["Date"].append(df_test["Date"])
    del df_train["Date"]
    del df_test["Date"]
    ols_model = get_ols_model(df_train, col=col)
    df_train["model"] = ols_model.predict(sm.add_constant(df_train.loc[:, df_train.columns!="Close"]).to_numpy())
    df_test["forecast"] = ols_model.predict(sm.add_constant(df_test.loc[:, df_test.columns!="Close"]).to_numpy())
    df = df_train.append(df_test)
    df["Date"] = dates
    df["ts"] = df[col]
    utils.plot_forecast(df=df, title="Linear Regression Forecast", summary=summary)

def get_ols_model(train_df, col, with_y_int = True):
    values = train_df[col].to_numpy()
    X = train_df.loc[:, train_df.columns!=col]
    if with_y_int:
        X = sm.add_constant((X))
    X = X.to_numpy()
    return sm.OLS(values, X).fit()
