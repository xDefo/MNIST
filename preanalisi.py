from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import math

def create_database():
  train=np.loadtxt("train.txt")
  print(np.shape(train))
  x_train=np.empty([60000,9])
  y_train=np.empty([60000])
  print(np.shape(x_train))
  idx=0
  for i in range(60000):
    y_train[i]=train[idx]
    idx+=1
    for j in range(9):
      x_train[i,j]=train[idx]
      idx+=1

  train=np.loadtxt("test.txt")
  x_test=np.empty([10000,9])
  y_test=np.empty([10000])
  idx=0
  for i in range(10000):
    y_test[i]=train[idx]
    idx+=1
    for j in range(9):
      x_test[i,j]=train[idx]
      idx+=1

  return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test=create_database()

mean=np.empty([10,9])
std=np.empty([10,9])
for j in range(10):
  indici=np.where(y_train==j)
  num=x_train[indici]

  for i in range(9):
    mean[j,i]=np.mean(num[:,i])
    std[j,i]=np.std(num[:,i])


angolo=np.arange(0,2*math.pi,0.001)

print(angolo)
color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
fig,ax=plt.subplots()
for i in range(9):
  x=mean[:,i]
  y=np.arange(0,10,1)

  ax.errorbar(x,y,xerr=std[j,i], capsize=10,capthick=1,color=color[i],label="feature{n}".format(n=i))

ax.set_xlabel("Media feature")
ax.set_yticks(np.arange(-1,10,1))
ax.set_ylabel("Numero")
ax.legend()

plt.show()
