import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

# --- 1. Load and preprocess the data ---
df = pd.read_csv("BrentOilPrices.csv")  # replace with actual path
# Parse date with day-month-year like '20-May-87'
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'Price'])
df = df.sort_values('Date').reset_index(drop=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# Optional: forward/backfill if small missing gaps
df['Price'] = df['Price'].interpolate(method='time')

# --- 2. Transformations ---
df['LogPrice'] = np.log(df['Price'])
df['LogReturn'] = df['LogPrice'].diff()  # approximate continuously compounded return
df = df.dropna(subset=['LogReturn'])

# --- 3. Basic EDA ---
print("Summary statistics on price:")
print(df['Price'].describe())

print("\nSummary statistics on log returns:")
print(df['LogReturn'].describe())

# Time series plots
plt.figure()
plt.plot(df['Date'], df['Price'])
plt.title("Brent Oil Price (USD)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df['Date'], df['LogReturn'])
plt.title("Brent Oil Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.tight_layout()
plt.show()

# Rolling statistics (window example: 60 trading days ~ quarter)
rolling_mean = df['LogReturn'].rolling(window=60).mean()
rolling_std = df['LogReturn'].rolling(window=60).std()

plt.figure()
plt.plot(df['Date'], rolling_mean, label='Rolling Mean (60d)')
plt.plot(df['Date'], rolling_std, label='Rolling Std (60d)')
plt.title("Rolling statistics of Log Returns")
plt.legend()
plt.tight_layout()
plt.show()

# ACF/PACF of log returns to check dependencies
plt.figure()
plot_acf(df['LogReturn'].dropna(), lags=40, zero=False)
plt.title("ACF of Log Returns")
plt.tight_layout()
plt.show()

plt.figure()
plot_pacf(df['LogReturn'].dropna(), lags=40, zero=False, method='ywm')
plt.title("PACF of Log Returns")
plt.tight_layout()
plt.show()

# --- 4. Stationarity tests ---
def adf_test(series, title=''):
    print(f"Augmented Dickey-Fuller Test: {title}")
    adf_res = ts.adfuller(series, autolag='AIC')
    output = {
        'Test Statistic': adf_res[0],
        'p-value': adf_res[1],
        'Lags Used': adf_res[2],
        'Number of Observations': adf_res[3]
    }
    for k, v in output.items():
        print(f"{k}: {v}")
    print("Critical Values:")
    for key, val in adf_res[4].items():
        print(f"   {key}: {val}")
    print()

def kpss_test(series, title=''):
    print(f"KPSS Test: {title}")
    kpss_res = ts.kpss(series, regression='c', nlags="auto")
    output = {
        'Test Statistic': kpss_res[0],
        'p-value': kpss_res[1],
        'Lags Used': kpss_res[2]
    }
    for k, v in output.items():
        print(f"{k}: {v}")
    print("Critical Values:")
    for key, val in kpss_res[3].items():
        print(f"   {key}: {val}")
    print()

# Test stationarity on log price and log return
adf_test(df['LogPrice'].dropna(), title='Log Price')
kpss_test(df['LogPrice'].dropna(), title='Log Price')
adf_test(df['LogReturn'].dropna(), title='Log Return')
kpss_test(df['LogReturn'].dropna(), title='Log Return')

# --- 5. Build event dataset (example) ---
events = [
    {"Event": "Iran nuclear deal sanction lift", "Date": "2015-07-14", "Type": "Supply increase", "Description": "JCPOA lifts sanctions on Iran, increasing crude supply."},
    {"Event": "OPEC+ output cut agreement", "Date": "2016-11-30", "Type": "Supply cut", "Description": "OPEC and allies agree to cut production to stabilize prices."},
    {"Event": "OPEC+ extension of cuts (2018)", "Date": "2018-01-01", "Type": "Supply management", "Description": "Continuation/extension of production cuts to clear glut."},
    {"Event": "Abqaiq–Khurais attack", "Date": "2019-09-14", "Type": "Supply disruption", "Description": "Attack on Saudi oil facilities causing temporary loss of output."},
    {"Event": "Russia-Saudi price war and COVID demand collapse", "Date": "2020-03-01", "Type": "Oversupply + demand shock", "Description": "Failed cut coordination and pandemic-caused demand crash."},
    {"Event": "Historic OPEC+ coordinated cut", "Date": "2020-04-12", "Type": "Supply cut", "Description": "Major production cuts agreed to stabilize collapsing prices."},
    {"Event": "Post-COVID demand rebound/global energy crisis", "Date": "2021-09-01", "Type": "Demand surge", "Description": "Recovery leads to tight market and price surge."},
    {"Event": "Russia invades Ukraine", "Date": "2022-02-24", "Type": "Geopolitical supply risk", "Description": "War and sanctions cause uncertainty and structural effects on supply/pricing."},
    {"Event": "OPEC+ surprise cuts amid recession concern", "Date": "2022-10-05", "Type": "Supply cut", "Description": "Steep production cuts to support prices."},
    {"Event": "OPEC+ voluntary cuts expansion (late 2023)", "Date": "2023-12-01", "Type": "Supply management", "Description": "Expanded voluntary cuts into 2024."},
]

events_df = pd.DataFrame(events)
events_df['Date'] = pd.to_datetime(events_df['Date'])
# Save to CSV for downstream use
events_df.to_csv("Brent_Event_List_Task1.csv", index=False)

print("Event dataset sample:")
print(events_df.head())
