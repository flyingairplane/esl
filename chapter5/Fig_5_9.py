##figure 5.9
##2020/09/24
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#参考维基百科https://en.wikipedia.org/wiki/Smoothing_spline#cite_note-Rodriguez-6
# 中的“Derivation of the cubic smoothing spline”部分
def constructA(knots):
    num=len(knots)
    delta=np.zeros((num-2,num))
    W=np.zeros((num-2,num-2))
    h=np.diff(knots)
    for i in range(0,num-2):
        delta[i,i]=1.0/h[i]
        delta[i,i+1]=-1.0/h[i]-1/h[i+1]
        delta[i,i+2]=1.0/h[i+1]
    for i in range(0,num-2):
        W[i,i]=(h[i]+h[i+1])/3
    for i in range(1,num-2):
        W[i-1,i]=h[i]/6
        W[i,i-1]=h[i]/6
    return delta,W

#一个简单的求lambda的函数
def compute_lambda(K,df):
    lam=0
    step=10
    lastsign=1
    sign=1
    tol=1e-3
    Slam=np.trace(np.linalg.inv(np.eye(K.shape[0])))
    while np.abs(Slam-df)>tol:
        if Slam>df:
            sign=1
            if sign==lastsign:
                lam=lam+step
            else:
                step=step/2
                lam=lam+step
            lastsign=sign
        else:
            sign=-1
            if sign!=lastsign:
                step=step/2
                lam=lam-step
            else:
                lam=lam-step
            lastsign=sign
        Slam = np.trace(np.linalg.inv(np.eye(K.shape[0]) + lam * K))
    return lam

dftest=np.linspace(2,15,num=20)
#仿真次数，用于求EPE和CV
simnum=200
#生成[0,1]区间内不重复的100个数
Nlen=100
EPE=np.zeros(shape=((len(dftest),)))
CV=np.zeros(shape=((len(dftest),)))
for i in np.arange(0,len(dftest)):
    df=dftest[i]
    print('df:'+str(df))
    for j in np.arange(0,simnum):
        x=np.random.choice(np.arange(0,1000),size=Nlen,replace=False)/1000
        x.sort()
        y=np.sin(12*(x+0.2))/(x+0.2)
        observedy=y+np.random.normal(size=((Nlen,)))
        predictedy = y + np.random.normal(size=((Nlen,)))

        knots = x
        delta, W = constructA(knots)
        K = delta.T.dot(np.linalg.inv(W)).dot(delta)
        lam = compute_lambda(K, df)
        Slam = np.linalg.inv(np.eye(K.shape[0]) + lam * K)
        f = Slam.dot(np.array(observedy).reshape((len(observedy), 1)))

        EPE[i]=EPE[i]+np.mean(np.square(y-f[:,0]))#MSE
        CV[i]=CV[i]+np.mean(np.square((observedy-f[:,0])/(1-np.diag(Slam))))
EPE=EPE/simnum+1#\sigma^2+MSE
CV=CV/simnum

fig, axes = plt.subplots(2,2)
axes[0][0].plot(dftest,EPE,color='b',marker='o',markersize=2,label='EPE')
axes[0][0].plot(dftest,CV,color='r',marker='o',markersize=2,label='CV')
axes[0][0].set_title('Cross-Validation')
axes[0][0].set_xlabel('df')
axes[0][0].legend(loc='upper right')


x=np.linspace(0,1,100)
y=np.sin(12*(x+0.2))/(x+0.2)

df=5
knots=x
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
Slam=np.linalg.inv(np.eye(K.shape[0]) + lam * K)
f=Slam.dot(np.array(observedy).reshape((len(observedy),1)))
std=np.sqrt(np.diag(Slam.dot(Slam.T)))

axes[0][1].scatter(x,observedy,marker='o',s=10)
axes[0][1].plot(x,y,color='b',label='true function')
axes[0][1].plot(x,f[:,0],color='r',label='fitted function')
axes[0][1].plot(x,f[:,0]+2*std,color='g',linestyle='--')
axes[0][1].plot(x,f[:,0]-2*std,color='g',linestyle='--')
axes[0][1].set_xlabel('x')
axes[0][1].set_ylabel('y')
axes[0][1].set_title('df='+str(df))
axes[0][1].legend(loc='upper right')

df=9
knots=x
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
Slam=np.linalg.inv(np.eye(K.shape[0]) + lam * K)
f=Slam.dot(np.array(observedy).reshape((len(observedy),1)))
std=np.sqrt(np.diag(Slam.dot(Slam.T)))

axes[1][0].scatter(x,observedy,marker='o',s=10)
axes[1][0].plot(x,y,color='b',label='true function')
axes[1][0].plot(x,f[:,0],color='r',label='fitted function')
axes[1][0].plot(x,f[:,0]+2*std,color='g',linestyle='--')
axes[1][0].plot(x,f[:,0]-2*std,color='g',linestyle='--')
axes[1][0].set_xlabel('x')
axes[1][0].set_ylabel('y')
axes[1][0].set_title('df='+str(df))
axes[1][0].legend(loc='upper right')

df=15
knots=x
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
Slam=np.linalg.inv(np.eye(K.shape[0]) + lam * K)
f=Slam.dot(np.array(observedy).reshape((len(observedy),1)))
std=np.sqrt(np.diag(Slam.dot(Slam.T)))

axes[1][1].scatter(x,observedy,marker='o',s=10)
axes[1][1].plot(x,y,color='b',label='true function')
axes[1][1].plot(x,f[:,0],color='r',label='fitted function')
axes[1][1].plot(x,f[:,0]+2*std,color='g',linestyle='--')
axes[1][1].plot(x,f[:,0]-2*std,color='g',linestyle='--')
axes[1][1].set_xlabel('x')
axes[1][1].set_ylabel('y')
axes[1][1].set_title('df='+str(df))
axes[1][1].legend(loc='upper right')

fig.tight_layout(pad=0,w_pad=1,h_pad=1)
plt.show()



# #特征值
# fig, ax = plt.subplots()
# ax.plot(np.arange(0,25),d1[0:25],color='b',marker='o',label='df=5')
# ax.plot(np.arange(0,25),d2[0:25],color='r',marker='o',label='df=11')
# ax.set_xlabel('Order')
# ax.set_ylabel('Eigenvalues')
# ax.legend(loc='upper right')
# plt.show()

# #特征向量
# fig,axes= plt.subplots(3, 2)
# for i in np.arange(0,6):
#     ax = fig.add_subplot(3,2,i+1)
#     ax.plot(ageunique,u1[:,i])
# fig.suptitle('Eigenvectos vs input')
# plt.show()

# #smoother matrix的热力图
# plt.imshow(np.log10(S1+1))
# plt.colorbar()
# plt.show()
#
# #每行smoother matrix
# rows=[12,25,50,75,100,115]
# fig,axes= plt.subplots(3, 2)
# for i in np.arange(0,6):
#     ax = fig.add_subplot(3,2,i+1)
#     ax.plot(ageunique,S1[rows[i],:],label='row='+str(rows[i]))
#     ax.plot(np.linspace(np.min(ageunique),np.max(ageunique),100),np.zeros((100,)),linestyle='--',color='k')
#     ax.legend(loc='upper right')
# fig.suptitle('Equivalent Kernels')
# plt.show()
