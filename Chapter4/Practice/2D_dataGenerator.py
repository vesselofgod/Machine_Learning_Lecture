import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(seed=1)
N=100
K=3
T3=np.zeros((N,3),dtype=np.uint8)
T2=np.zeros((N,2),dtype=np.uint8)
X=np.zeros((N,2))

X_range0=[-3,3]
X_range1=[-3,3]

Mu=np.array([[-.5,-.5],[.5,1.0],[1,-.5]])
Sig = np.array([[.7,.7],[.8,.3],[.3,.8]])
Pi=np.array([0.4,0.8,1])
for n in range(N):
    wk=np.random.rand()
    for k in range(K):
        if wk<Pi[k]:
            T3[n,k]=1
            break
    for k in range(2):
        X[n,k] = (np.random.randn() * Sig[T3[n, :] == 1,k] + Mu[T3[n, :] == 1,k])
    T2[:,0]=T3[:,0]
    T2[:,1]=T3[:,1] | T3[:,2]


def logistic2(x0,x1,w):
    y=1/(1+np.exp(-(w[0]*x0 + w[1]*x1 + w[2])))
    return y
def show_data2(x,t):
    wk,k = t.shape
    c=[[.5,.5,.5],[1,1,1],[0,0,0]]
    for k in range(K-1):
        plt.plot( x[(t[: , k] == 1,0)],x[(t[:,k] == 1,1)], linestyle = 'none', markeredgecolor = 'black', marker = 'o', color = c[k], alpha = 0.8)
        plt.grid(True)

def show_contour_logistic2(w):
    xn=30
    x0=np.linspace(X_range0[0],X_range0[1],xn)
    x1=np.linspace(X_range1[0],X_range1[1],xn)
    xx0,xx1=np.meshgrid(x0,x1)
    y=logistic2(xx0,xx1,w)
    cont=plt.contour(xx0,xx1,y,levels=(0.2,0.5,0.8),colors=['k','cornflowerblue','k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)

def cee_logistic2(w,x,t):
    X_n=x.shape[0]
    y=logistic2(x[:,0],x[:,1],w)
    cee=0
    ##내용을 채워넣으시오
    return cee
def dcee_logistic2(w,x,t):
    X_n=x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    ##내용을 채워넣으시오
    return dcee

def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x,t), jac=dcee_logistic2, method="CG")
    return res.x

#main
plt.figure(1,figsize=(4,4))
show_data2(X,T2)
plt.xlim(X_range0)
plt.ylim(X_range1)

W_init=[-1,0,0]
W=fit_logistic2(W_init,X,T2)
print(W)
cee=cee_logistic2(W,X,T2)
show_contour_logistic2(W)
plt.show()