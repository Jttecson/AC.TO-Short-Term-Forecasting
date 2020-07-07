import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plots model, forecast, and residual plots for the
# given df model.
# Should have columns:
# ts        -   time series data
# model     -   model fit on training data
# forecast  -   forecasted values
# Date      -   dates
def plot_forecast(df, title, summary=True):
    ## residuals
    df["residuals"] = df["ts"] - df["model"]
    df["error"] = df["ts"] - df["forecast"]
    df["error_pct"] = df["error"] / df["ts"]

    ## kpi
    residuals_mean = df["residuals"].mean()
    residuals_std = df["residuals"].std()
    error_mean = df["error"].mean()
    error_std = df["error"].std()
    mae = df["error"].apply(lambda x: np.abs(x)).mean()
    mape = df["error_pct"].apply(lambda x: np.abs(x)).mean()
    mse = df["error"].apply(lambda x: x ** 2).mean()
    rmse = np.sqrt(mse)

    # # intervals
    # df["conf_int_low"] = df["forecast"] - 1.96 * residuals_std
    # df["conf_int_up"] = df["forecast"] + 1.96 * residuals_std
    # df["pred_int_low"] = df["forecast"] - 1.96 * error_std
    # df["pred_int_up"] = df["forecast"] + 1.96 * error_std

    # plot
    fig = plt.figure(figsize=(10,6))
    fig.suptitle(title, fontsize=20)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Model fit
    df[pd.notnull(df["model"])][["ts", "model","Date"]].plot(color=["black", "green"], x="Date", y=["ts", "model"], grid=True, ax=ax1)
    ax1.set(ylabel="Close Price (CAD)")
    ax1.set_title("Model Fit", size=15)
    ax1.legend(bbox_to_anchor=(0.65, 0.6, 0.2, 0.15),loc=3)

    # Forecast with conf intervals
    df[pd.isnull(df["model"])][["ts", "forecast"]].plot(color=["black", "red"], grid=True, ax=ax2)
    ax2.set(xlabel=None)
    ax2.set_title("Forecast", size=15)
    ax2.legend(bbox_to_anchor=(0.65, 0.6, 0.2, 0.15),loc=3)

    # Residuals
    df.plot(ax=ax3, c="green", kind="scatter", x="Date", y="residuals", grid=True, label="Model")
    df.plot(ax=ax3, c="red", kind="scatter", x="Date", y="error", grid=True, label="Forecast")
    ax3.set(xlabel=None)
    ax3.set(ylabel=None)
    ax3.set_title("Residuals", size=15)
    ax3.legend(bbox_to_anchor=(0.65, 0.6, 0.2, 0.15),loc=3)

    # Residual distribution
    df[["residuals", "error"]].plot(ax=ax4, color=["green", "red"], kind='kde', grid=True)
    ax4.set(ylabel=None)
    ax4.set_title("Residual Distribution", size=15)
    ax4.legend(bbox_to_anchor=(0.65, 0.6, 0.2, 0.15),loc=3)
    fig.autofmt_xdate()
    print("Training --> Residuals mean:", (residuals_mean), " | std:", (residuals_std))
    print("Test --> Error mean:", (error_mean), " | std:", (error_std), " | mae:", (mae),
          " | mape:", (mape * 100), "%  | mse:", (mse), " | rmse:", (rmse))