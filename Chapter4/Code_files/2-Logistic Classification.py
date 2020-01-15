import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(seed=0)
X_min=0
X_max=2.5
X_n=30
X_col = ['cornflowerblue','gray']
X=np.zeros(X_n) #input data
T=np.zeros(X_n, dtype=np.uint8) #object data

Dist_s=[0.4,0.8] #distributions's start point
Dist_w=[0.8, 1.6] #distribution's width
Pi=0.5
for n in range(X_n):
    wk=np.random.rand()
    T[n] = 0*(wk < Pi) + 1*(wk >= Pi)
    X[n] = np.random.rand()*Dist_w[T[n]] + Dist_s[T[n]]


def show_data1(x,t):
    #show each datas to point
    K=np.max(t)+1
    for k in range(K):
        plt.plot(x[t==k],t[t==k],X_col[k],alpha=0.5, linestyle='none',marker='o')
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(X_min,X_max)
        plt.yticks([0,1])

def logistic(x,w):
    y=1/(1+np.exp(-(w[0]*x+w[1])))
    return y
def show_logistic(w):
    #show sigmoid graph
    xb=np.linspace(X_min,X_max,100)
    y=logistic(xb,w)
    plt.plot(xb,y)
    #show decision boundary
    i=np.min(np.where(y>0.5))
    B=(xb[i-1]+xb[i])/2
    plt.plot([B,B],[-.5,1.5],color='k',linestyle='--')
    plt.grid(True)
    return B
def cee_logistic(w,x,t):
    #calc avg cross entropy error
    y=logistic(x,w)
    cee=0
    for n in range(len(y)):
        cee = cee -(t[n]*np.log(y[n]) + (1-t[n]) * np.log(1-y[n]))
    cee = cee / X_n
    return cee
def dcee_logistic(w,x,t):
    y=logistic(x,w)
    dcee=np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0]+(y[n]-t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee/X_n
    return dcee

def fit_logistic(w_init,x,t):
    #calculate weight using gradient decent
    res1=minimize(cee_logistic,w_init,args=(x,t), jac=dcee_logistic, method='CG')
    return res1

#main
plt.figure(1,figsize=(3,3))
#Linear regression initial setting
W_init=[1,-1]
#calculate and print weight
W=fit_logistic(W_init,X,T)
print("wo,w1 = ", W.x)
B=show_logistic((W.x[0],W.x[1]))
show_data1(X,T)
plt.ylim(-.5,1.5)
plt.xlim(X_min,X_max)
#calculate Avg cross entropy error
cee=cee_logistic((W.x[0],W.x[1]),X,T)
print("Cross Entropy Error = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(B))
plt.show()
