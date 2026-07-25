import numpy as np

np.random.seed(10)

actual_mean=5
actual_var=4

data=np.random.normal(actual_mean,np.sqrt(actual_var),1000)

mle_mean=np.mean(data)
mle_var=np.var(data)

print("Actual Mean:",actual_mean)
print("Estimated Mean:",mle_mean)

print("Actual Variance:",actual_var)
print("Estimated Variance:",mle_var)
