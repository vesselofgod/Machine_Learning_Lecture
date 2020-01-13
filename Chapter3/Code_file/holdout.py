import numpy as np
import matplotlib.pyplot as plt

outfile=np.load('ch3_data.npz')
X=outfile['X']
X_min=outfile['X_min']
X_max=outfile['X_max']
X_n=outfile['X_n']
T=outfile['T']

def gauss(x, mu, s):
    #가우스 함수
    return np.exp(-(x-mu)**2/(2*s**2))

def gauss_func(w,x):
    #return linear basis function model
    m=len(w)-1
    mu=np.linspace(5,30,m)
    s = mu[1] - mu[0]
    y=np.zeros_like(x)
    for j in range(m):
        y=y+w[j]*gauss(x,mu[j],s)
    y=y+w[m]
    return y

def mse_gauss_func(x,t,w):
    #평균제곱오차 구해주는 함수
    y=gauss_func(w,x)
    mse=np.mean((y-t)**2)
    return mse

def fit_gauss_func(x,t,m):
    #linear basis function model의 solution을 return
    mu=np.linspace(5,30,m)
    s=mu[1]-mu[0]
    n=x.shape[0]
    psi=np.ones((n,m+1))
    for j in range(m):
        psi[:,j] = gauss(x,mu[j],s)
    psi_T = np.transpose(psi)

    b=np.linalg.inv(psi_T.dot(psi))
    c=b.dot(psi_T)
    w=c.dot(t)
    return w

def show_gauss_func(w):
    xb=np.linspace(X_min,X_max,100)
    y=gauss_func(w,xb)
    plt.plot(xb,y,c=[.5,.5,.5],lw=4)

#split test data and training data
X_test=X[:int(X_n / 4 + 1)]
T_test=T[:int(X_n / 4 + 1)]
X_train = X[int(X_n / 4 + 1):]
T_train = T[int(X_n / 4 + 1):]

#main

plt.figure(figsize=(10,2.5))

plt.subplots_adjust(wspace=0.3)
M=[2,4,7,9]
for i in range(len(M)):
    plt.subplot(1,len(M),i+1)
    W=fit_gauss_func(X_train,T_train,M[i])
    show_gauss_func(W)
    plt.plot(X_train,T_train, marker='o', linestyle='None', color='white', markeredgecolor='black', label='training')
    plt.plot(X_test, T_test, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black', label='test')
    plt.legend(loc='lower right', fontsize=10, numpoints=1)
    plt.xlim(X_min,X_max)
    plt.ylim(130,180)
    plt.grid(True)
    mse=mse_gauss_func(X_test,T_test,W)
    plt.title("M={0:d}, SD={1:.1f}".format(M[i],np.sqrt(mse)))

plt.show()
