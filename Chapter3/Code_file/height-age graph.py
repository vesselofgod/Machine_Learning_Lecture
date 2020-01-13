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


'''
M을 4로 했을 때의 gauss basis function
가우스 함수의 중심(Avg)는 5~30에서 균등하게 배치한 후,
표준편차를 8.33으로 두었다.
'''


M=4
plt.figure(figsize=(4,4))
mu=np.linspace(5,30,M)
s=mu[1]-mu[0]
xb=np.linspace(X_min,X_max,100)
for j in range(M):
    y=gauss(xb,mu[j],s)
    plt.plot(xb,y,linewidth=3)
plt.grid(True)
plt.xlim(X_min,X_max)
plt.ylim(0,1.2)
plt.show()
