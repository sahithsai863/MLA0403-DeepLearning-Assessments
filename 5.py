import numpy as np

x=np.array([1,2,3])

w=np.array([0.5,0.2,0.8])

z=np.dot(x,w)

sigmoid=1/(1+np.exp(-z))

relu=max(0,z)

print("Weighted Sum:",z)
print("Sigmoid:",sigmoid)
print("ReLU:",relu)
