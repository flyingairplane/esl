##figure 5.7
##2020/09/23
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
    step=100
    lastsign=1
    sign=1
    tol=1e-4
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

currentfolder='D:\\Program\\python\\esl\\chapter5\\'
totaldata=pd.read_csv(currentfolder+'LAozone.data.txt',delimiter=',',header=0,index_col=False)
dealdata=totaldata[['dpg','ozone']]

ageunique,counts=np.unique(dealdata.dpg,return_counts=True)
ageunique.sort()
#重复年龄加权
weights=np.diag(counts[np.argsort(ageunique)])
yunique=[np.mean(dealdata.ozone[dealdata.dpg==age]) for age in ageunique]

df=5
knots=ageunique
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
# #拟合结果
# f=np.linalg.inv(np.eye(K.shape[0]) + lam * K).dot(np.array(yunique).reshape((len(yunique),1)))
#加权拟合结果
fw1=np.linalg.inv(weights + lam * K).dot(weights).dot(np.array(yunique).reshape((len(yunique),1)))
#特征值
S1=np.linalg.inv(np.eye(K.shape[0]) + lam * K)
u1,d1,_=np.linalg.svd(S1)

df=11
knots=ageunique
delta,W=constructA(knots)
K=delta.T.dot(np.linalg.inv(W)).dot(delta)
lam=compute_lambda(K,df)
# #拟合结果
# f=np.linalg.inv(np.eye(K.shape[0]) + lam * K).dot(np.array(yunique).reshape((len(yunique),1)))
#加权拟合结果
fw2=np.linalg.inv(weights + lam * K).dot(weights).dot(np.array(yunique).reshape((len(yunique),1)))
#特征值
S2=np.linalg.inv(np.eye(K.shape[0]) + lam * K)
u2,d2,_=np.linalg.svd(S2)

# fig, ax = plt.subplots()
# ax.scatter(dealdata.dpg,dealdata.ozone,marker='o',s=10)
# ax.plot(ageunique,fw1[:,0],color='b',label='df=5')
# ax.plot(ageunique,fw2[:,0],color='r',label='df=11')
# ax.set_xlabel('Daggot Pressure Gradient')
# ax.set_ylabel('Ozone Concentration')
# ax.legend(loc='upper right')
# plt.show()


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

#smoother matrix的热力图
plt.imshow(np.log10(S1+1))
plt.colorbar()
plt.show()

#每行smoother matrix
rows=[12,25,50,75,100,115]
fig,axes= plt.subplots(3, 2)
for i in np.arange(0,6):
    ax = fig.add_subplot(3,2,i+1)
    ax.plot(ageunique,S1[rows[i],:],label='row='+str(rows[i]))
    ax.plot(np.linspace(np.min(ageunique),np.max(ageunique),100),np.zeros((100,)),linestyle='--',color='k')
    ax.legend(loc='upper right')
fig.suptitle('Equivalent Kernels')
plt.show()
