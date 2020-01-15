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

def show_data2(x,t):
    wk,k = t.shape
    c=[[.5,.5,.5],[1,1,1],[0,0,0]]
    for k in range(K):
        plt.plot( x[(t[: , k] == 1,0)],x[(t[:,k] == 1,1)], linestyle = 'none', markeredgecolor = 'black', marker = 'o', color = c[k], alpha = 0.8)
        plt.grid(True)

def logistic3(x0,x1,w):
    #3개의 class를 구분해내기 위한 logistic regression model
    #*행렬연산을 수행함.
    K=3
    w=w.reshape((3,3))
    n=len(x1)
    y=np.zeros((n,K))
    #빈칸을 채우시오
    return y

def cee_logistic3(w,x,t):
    #교차 엔트로피 오차를 구하는 함수.
    X_n=x.shape[0]
    y=logistic3(x[:,0], x[:,1],w)
    cee = 0
    N,K=y.shape
    #빈칸을 채우시오
    return cee

def dcee_logistic3(w,x,t):
    #교차 엔트로피 오차를 미분하여 오차를 최소화하게끔 해주는 함수
    X_n=x.shape[0]
    y=logistic3(x[:,0],x[:,1],w)
    dcee = np.zeros((3,3))
    N,K=y.shape
    #빈칸을 채우시오
    return dcee.reshape(-1)

def fit_logistic3(w_init,x,t):
    res = minimize(cee_logistic3, w_init, args=(x,t), jac=dcee_logistic3,  method="CG")
    return res.x

def show_contour_logistic3(w):
    xn=30
    x0=np.linspace(X_range0[0],X_range0[1],xn)
    x1=np.linspace(X_range1[0],X_range1[1],xn)

    xx0,xx1=np.meshgrid(x0,x1)
    y=np.zeros((xn,xn,3))

    for i in range(xn):
        wk=logistic3(xx0[:,i],xx1[:,i],w)
        for j in range(3):
            y[:,i,j]=wk[:,j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:,:,j], levels=(0.5, 0.8), colors=['cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)

#main
W_init=np.zeros((3,3))
W=fit_logistic3(W_init, X,T3)
print(np.round(W.reshape((3,3)),2))
cee=cee_logistic3(W,X,T3)
print("CEE = {0:.2f}".format(cee))

plt.figure(figsize=(4,4))
show_data2(X,T3)
show_contour_logistic3(W)
plt.show()
