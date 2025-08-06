import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your cleaned data from Task 1
df = pd.read_csv("BrentOilPrices.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'Price'])
df = df.sort_values('Date').reset_index(drop=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# Compute log returns
df['LogPrice'] = np.log(df['Price'])
df['LogReturn'] = df['LogPrice'].diff()
df = df.dropna(subset=['LogReturn']).reset_index(drop=True)

returns = df['LogReturn'].values
dates = df['Date'].values
n = len(returns)
import pymc3 as pm

with pm.Model() as model_cp:
    # Prior for the change point τ (uniformly distributed in the time range)
    tau = pm.DiscreteUniform("tau", lower=0, upper=n)

    # Priors for the means before and after the change point
    mu1 = pm.Normal("mu1", mu=0, sigma=0.05)
    mu2 = pm.Normal("mu2", mu=0, sigma=0.05)

    # Shared standard deviation
    sigma = pm.HalfNormal("sigma", sigma=0.05)

    # Deterministic piecewise mean
    mu = pm.math.switch(tau >= np.arange(n), mu1, mu2)

    # Likelihood
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=returns)

    # Inference
    trace_cp = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True, random_seed=42)
import arviz as az

# Summary of posterior
az.summary(trace_cp, var_names=["mu1", "mu2", "sigma", "tau"])

# Posterior plot for tau (change point index)
az.plot_posterior(trace_cp, var_names=["tau"])
plt.title("Posterior distribution of Change Point τ")
plt.show()

# Convert τ from index to date
tau_samples = trace_cp.posterior["tau"].values.flatten()
tau_mean = int(tau_samples.mean())
tau_date = dates[tau_mean]
print(f"Estimated change point (mean): {tau_date}")
plt.figure(figsize=(12, 5))
plt.plot(dates, returns, label='Log Return', alpha=0.7)
plt.axvline(tau_date, color='red', linestyle='--', label=f'Estimated Change Point: {tau_date}')
plt.title("Brent Crude Log Returns with Detected Change Point")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.tight_layout()
plt.show()
events_df = pd.read_csv("Brent_Event_List_Task1.csv")
events_df['Date'] = pd.to_datetime(events_df['Date'])

# Compare to closest event
events_df['Days From Change Point'] = (events_df['Date'] - tau_date).dt.days.abs()
matched_event = events_df.sort_values('Days From Change Point').iloc[0]
print("Closest Event to Detected Change Point:")
print(matched_event)
