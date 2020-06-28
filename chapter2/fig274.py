##fig2.7的第四幅图
##
##Date:2020/06/26
import numpy as np
import matplotlib.pyplot as plt
dimension=range(1,11)
trainsize=100#训练样本集大小
trainnum=100#训练样本集数量
varian=np.zeros(shape=(len(dimension),1))#方差
sqbias=np.zeros(shape=(len(dimension),1))#偏差的平方
ytrain=np.zeros(shape=(trainsize,1))
ytest=np.zeros(shape=(trainnum,1))
for i in range(len(dimension)):
    tdim=dimension[i]
    xtest=np.zeros(shape=(1,tdim))
    for j in range(trainnum):#生成单个训练样本集
        xtrain=np.random.random(size=(trainsize,tdim))*2-1#生成训练样本
        ytrain=np.exp(-8*sum((xtrain**2).T))#使用的函数
        dist=np.zeros(shape=(trainsize,1))
        for k in range(trainsize):
            dist[k]=sum((xtrain[k,:]**2).T)
        onennx=dist.argmin()#最近邻的坐标
        ytest[j]=ytrain[onennx]
    varian[i]=np.mean((ytest-np.mean(ytest))**2)
    sqbias[i]=(np.mean(ytest)-1)**2#f(0)=1
fig=plt.figure()
ax2=fig.add_subplot(1,1,1)
ax2.plot(dimension,varian,'bo-',label='variance')
ax2.plot(dimension,sqbias,'ro-',label='square bias')
ax2.plot(dimension,varian+sqbias,'ko-',label='mse')
plt.legend(loc='upper right')
plt.show()