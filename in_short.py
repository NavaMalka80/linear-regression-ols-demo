import numpy as np

# 1. creat data;
beta0 = 0.5
beta1 = 2

n = 100000
X = np.random.uniform(0, 10, n)

noise = np.random.normal(0, 1, n)

Y = beta0 + beta1 * X + noise

# 2. calculating the optimal line
X_mean = np.mean(X)
Y_mean = np.mean(Y)


beta1_hat = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)

beta0_hat = Y_mean - beta1_hat * X_mean