import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
beta0 = 0.5  # Intercept (true value)
beta1 = 2.0  # Slope (true value)
n = 100000
X = np.random.uniform(0, 10, n)  # Random X values, uniform distribution
noise = np.random.normal(0, 1, n)  # Noise, standard normal distribution
Y = beta0 + beta1 * X + noise  # Generate corresponding Y with noise

# Step 2: Calculate optimal OLS line parameters
X_mean = np.mean(X)
Y_mean = np.mean(Y)
beta1_hat = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
beta0_hat = Y_mean - beta1_hat * X_mean

# Print/log results for test/unit test file
output_log = []
def log(msg):
    print(msg)
    output_log.append(str(msg))

log(f'True beta0: {beta0}, True beta1: {beta1}')
log(f'Calculated beta0_hat: {beta0_hat:.3f}, beta1_hat: {beta1_hat:.3f}')

mse = np.mean((Y - (beta0_hat + beta1_hat * X))**2)
log(f'Calculated MSE: {mse:.3f}')

# Generate and save a plot (with a sample for readability)
idx = np.random.choice(n, size=200, replace=False)
plt.figure(figsize=(8,6))
plt.scatter(X[idx], Y[idx], color='blue', alpha=0.5, label='Data points')
# Plot the regression line
X_line = np.linspace(0, 10, 100)
Y_line = beta0_hat + beta1_hat * X_line
plt.plot(X_line, Y_line, color='red', label='OLS optimal line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('OLS Linear Fit to Data')
plt.legend()
plt.tight_layout()
plt.savefig('fit_plot.png')
plt.close()

# Save log outputs to a file
with open('output_log.txt', 'w', encoding='utf8') as f:
    for line in output_log:
        f.write(line+'\n')
