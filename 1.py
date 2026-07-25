import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5],dtype=float)
y = np.array([2,4,6,8,10],dtype=float)

w=0
b=0
lr=0.01
epochs=1000
losses=[]

n=len(x)

for i in range(epochs):
    y_pred=w*x+b

    dw=(-2/n)*np.sum(x*(y-y_pred))
    db=(-2/n)*np.sum(y-y_pred)

    w=w-lr*dw
    b=b-lr*db

    loss=np.mean((y-y_pred)**2)
    losses.append(loss)

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.show()

print("Weight:",w)
print("Bias:",b)
