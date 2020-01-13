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

def kfold_gauss_func(x,t,m,k):
    #K times split data and return each MSE
    n=x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)
    for i in range(0,k):
        #fmod method : return (n mod k)
        #0~k-1까지를 반복하는 n개의 list를 얻을 수 있음.
        x_train = x[np.fmod(range(n),k) != i]
        t_train = t[np.fmod(range(n),k) != i]
        x_test = x[np.fmod(range(n),k) == i]
        t_test = t[np.fmod(range(n),k) == i]

        #각각의 k개로 나뉘어진 데이터들을 이용해서 training과 test를 함.
        wm = fit_gauss_func(x_train,t_train,m)
        mse_train[i]=mse_gauss_func(x_train,t_train,wm)
        mse_test[i] = mse_gauss_func(x_test, t_test, wm)
    return mse_train,mse_test

#main

M=range(2,8)
K=16
Cv_Gauss_train = np.zeros((K,len(M)))
Cv_Gauss_test = np.zeros((K,len(M)))
for i in range(0,len(M)):
    Cv_Gauss_train[:, i],Cv_Gauss_test[:, i] =kfold_gauss_func(X,T,M[i],K)
    mean_Gauss_train = np.sqrt(np.mean(Cv_Gauss_train,axis=0))
    mean_Gauss_test = np.sqrt(np.mean(Cv_Gauss_test, axis=0))
plt.figure(figsize=(4,3))
plt.plot(M, mean_Gauss_train , marker='o', linestyle='-', color='k', markeredgecolor='w', label='training')
plt.plot(M, mean_Gauss_test , marker='o', linestyle='-', color='cornflowerblue', markeredgecolor='black', label='test')
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, 20)
plt.grid(True)
plt.show()