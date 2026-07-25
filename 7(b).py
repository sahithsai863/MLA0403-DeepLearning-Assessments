w=2
x=4
target=10
lr=0.01

prediction=w*x

error=target-prediction

gradient=-2*x*error

w_new=w-lr*gradient

print("Updated Weight:",w_new)
