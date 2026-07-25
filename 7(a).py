import numpy as np

x=np.array([1,2])

w=np.array([0.5,0.3])

b=0.1

z=np.dot(x,w)+b

output=1/(1+np.exp(-z))

print("Output:",output)
