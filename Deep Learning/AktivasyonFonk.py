import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
 
#aktivasyon fonksiyonları, activation functions
 
def step(x):
    return np.array(x > 0, dtype=np.int)
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def relu(x):
    return np.maximum(0, x)
 
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
 
def softplus(x):
    return np.log(1.0 + np.exp(x))
 
def tanh(x):
    return np.tanh(x)
 
def swish(x):
    #return x*sigmoid(x)
    return x*(1 / (1 + np.exp(-x)))
 
def prelu(x,alpha):
    a = []
    for item in x:
        if item < 0:
            a.append(alpha*item)
        else:
            a.append(item)
    return a
 
def elu(x,alpha):
    a = []
    for item in x:
        if item >= 0:
            a.append(item)
        else:
            a.append(alpha * (np.exp(item)-1))
    return a
 
x = np.arange(-5., 5., 0.1)
 
step=step(x)
sigmoid=sigmoid(x)
relu=relu(x)
softmax=softmax(x)
softplus=softplus(x)
tanh=tanh(x)
prelu=prelu(x,0.1)
elu=elu(x,1.0)
swish=swish(x)
 
#grafik çizimleri, plotting
 
plt.figure()
plt.xlabel("Girdiler")
plt.ylabel("Fonksiyon Çıktıları")
plt.grid(True)
plt.plot(x,step, label="Step", color='C0', lw=3)
plt.plot(x,sigmoid, label="Sigmoid", color='C1', lw=3)
plt.plot(x,relu, label="ReLU", color='C2', lw=3)
plt.plot(x,softmax, label="Softmax", color='C3', lw=3)
plt.plot(x,softplus, label="SoftPlus", color='C4', lw=3)
plt.plot(x,tanh, label="TanH", color='C5', lw=3)
plt.plot(x,prelu, label="PReLU", color='C6', lw=3)
plt.plot(x,elu, label="ELU", color='C8', lw=3)
plt.plot(x,swish, label="Swish", color='C9', lw=3)
plt.legend()
plt.show()