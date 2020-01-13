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

#main
M=4
plt.figure(figsize=(4,4))
W=fit_gauss_func(X,T,M)
show_gauss_func(W)
plt.plot(X,T,marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(X_min,X_max)
plt.grid(True)
mse=mse_gauss_func(X,T,W)
print('W='+str(np.round(W,1)))
print("SD={0:.2f} cm".format(np.sqrt(mse)))
plt.show()


