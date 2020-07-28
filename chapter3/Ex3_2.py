###esl第三章Ex 3.2
###
###Date:2020/07/27
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
N=40#样本数
tao=1#噪声方差
beta=[1.3965849,1.9407724,-0.1215529,0.1535441]
x1=np.linspace(-2,2,N)
x=np.zeros(shape=(N,4))
x[:,0]=1
x[:,1]=x1
x[:,2]=np.power(x1,2)
x[:,3]=np.power(x1,3)
y=x.dot(beta)+np.random.normal(size=(N,))
invXTX=np.linalg.inv(x.T.dot(x))
betahat=invXTX.dot(x.T).dot(y)
yhat=x.dot(betahat)

#second method
tao2=np.sum((y-yhat)**2)/(N-4)
#显著性临界点
sf=stats.chi2.isf(0.05,4)
plt.style.use('ggplot')
fig, ax = plt.subplots()
lw=2
#x.T*x=L*L.T
for i in range(0,50):
    #alpha=np.random.normal(size=(4,))
    alpha=np.random.random(size=(4,))
    alpha=alpha-np.mean(alpha)
    L=np.linalg.cholesky(x.T.dot(x))
    t0=(L.T.dot(alpha)).T.dot(L.T.dot(alpha))
    alpha = alpha/np.sqrt(t0/(sf*tao2))
    beta0=betahat-alpha
    print((beta0-betahat).T.dot(x.T).dot(x).dot(beta0-betahat))
    plt.plot(x1,x.dot(beta0),c='g',ls='-.')
# plt.plot(x1, yhat, c='k',ls='-')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

#first method
varu=np.zeros((N,))
for i in range(len(varu)):
    varu[i]=x[i,:].dot(invXTX).dot(x[i,:].T)

# plt.style.use('ggplot')
# fig, ax = plt.subplots()
lw=2
plt.plot(x1, yhat, c='k',ls='-')
plt.plot(x1,yhat+1.96*np.sqrt(varu),c='r',ls='-')
plt.plot(x1,yhat-1.96*np.sqrt(varu),c='r',ls='-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()