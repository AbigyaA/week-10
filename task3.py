import arviz as az
import numpy as np

# Extract posterior samples for τ
tau_samples = trace_cp.posterior['tau'].values.flatten()
mu1_samples = trace_cp.posterior['mu1'].values.flatten()
mu2_samples = trace_cp.posterior['mu2'].values.flatten()
sigma_samples = trace_cp.posterior['sigma'].values.flatten()

# Convert τ indices to actual dates
tau_dates = [dates[i] for i in tau_samples.astype(int)]
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.hist(tau_dates, bins=50, alpha=0.8, color='skyblue', edgecolor='black')
plt.title("Posterior Distribution of Change Point (τ)")
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Compute mean and 95% credible interval
tau_mean_idx = int(np.mean(tau_samples))
tau_mean_date = dates[tau_mean_idx]
ci_lower_idx = int(np.percentile(tau_samples, 2.5))
ci_upper_idx = int(np.percentile(tau_samples, 97.5))
ci_lower_date = dates[ci_lower_idx]
ci_upper_date = dates[ci_upper_idx]

print(f"Mean change point (τ): {tau_mean_date}")
print(f"95% credible interval: [{ci_lower_date}, {ci_upper_date}]")
events_df = pd.read_csv("Brent_Event_List_Task1.csv")
events_df['Date'] = pd.to_datetime(events_df['Date'])

plt.figure(figsize=(12, 6))
plt.hist(tau_dates, bins=50, alpha=0.8, color='skyblue', edgecolor='black')

# Mark mean and credible interval
plt.axvline(tau_mean_date, color='red', linestyle='--', label=f'Mean τ: {tau_mean_date.date()}')
plt.axvline(ci_lower_date, color='orange', linestyle=':', label='95% CI')
plt.axvline(ci_upper_date, color='orange', linestyle=':')

# Plot event lines
for i, row in events_df.iterrows():
    if ci_lower_date <= row['Date'] <= ci_upper_date:
        plt.axvline(row['Date'], color='black', linestyle='-', alpha=0.6)
        plt.text(row['Date'], 5, row['Event'], rotation=90, verticalalignment='center', fontsize=8)

plt.title("Posterior τ with Historical Events Overlay")
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
# Use mean τ to partition the series
tau_idx = tau_mean_idx

mu1 = np.mean(mu1_samples)
mu2 = np.mean(mu2_samples)

plt.figure(figsize=(14, 5))
plt.plot(dates, returns, label='Log Returns', alpha=0.6)
plt.axvline(tau_mean_date, color='red', linestyle='--', label=f'Change Point: {tau_mean_date.date()}')
plt.hlines(mu1, xmin=dates[0], xmax=dates[tau_idx], colors='green', linestyles='-', label=f'Mean before τ = {mu1:.5f}')
plt.hlines(mu2, xmin=dates[tau_idx], xmax=dates[-1], colors='blue', linestyles='-', label=f'Mean after τ = {mu2:.5f}')
plt.title("Log Returns with Estimated Means Before and After Change Point")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.legend()
plt.tight_layout()
plt.show()
az.plot_posterior(trace_cp, var_names=['mu1', 'mu2', 'sigma'], hdi_prob=0.95)
plt.tight_layout()
plt.show()
events_df['Days From Change Point'] = (events_df['Date'] - tau_mean_date).dt.days.abs()
closest_event = events_df.sort_values('Days From Change Point').iloc[0]

print("Closest Historical Event to τ:")
print(closest_event[['Event', 'Date', 'Type', 'Description']])
