import numpy as np

data=np.arange(100)

batch_size=20

for i in range(0,len(data),batch_size):
    batch=data[i:i+batch_size]
    print(batch)
