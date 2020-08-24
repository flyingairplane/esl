##figure 4.2
##2020/08/11
##by flyingairplane
import numpy as np
import matplotlib.pyplot as plt
means=np.array([[0,0],[1,1],[2,2]])
covtrain=np.eye(2)/10
trainnum=100#每类的样本数
classnum=3#类别数
xtrain=np.zeros(shape=(trainnum*classnum,2))
ytrain=np.zeros(shape=(trainnum*classnum,3))
for i in range(0,classnum):
    xtrain[i*trainnum:(i+1)*trainnum,:]=np.random.multivariate_normal(means[i],covtrain,(trainnum,))

#dummy
ytrain[0:trainnum,0]=1#red
ytrain[trainnum:2*trainnum,1]=1#blue
ytrain[2*trainnum:3*trainnum,2]=1#black

#original data
# fig=plt.figure()
# ax1=fig.add_subplot(1,1,1)
# ax1.scatter(xtrain[:,0],xtrain[:,1],marker="o",c="orange")
# plt.show()

#linear regression
xtrainadd=np.hstack((np.ones((trainnum*classnum,1)),xtrain))
ycal=xtrainadd.dot(np.linalg.inv(xtrainadd.T.dot(xtrainadd))).dot(xtrainadd.T).dot(ytrain)
ypredict=np.zeros(shape=(trainnum*classnum,))
for i in range(0,trainnum*classnum):
    ypredict[i]=np.argmax(ycal[i,:])

#classification results
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.scatter(xtrain[ypredict==0,0],xtrain[ypredict==0,1],marker="o",c="red")
ax1.scatter(xtrain[ypredict==1,0],xtrain[ypredict==1,1],marker="o",c="black")
ax1.scatter(xtrain[ypredict==2,0],xtrain[ypredict==2,1],marker="o",c="blue")
ax1.plot(np.linspace(0,2,100),2-np.linspace(0,2,100),'-.')

#y的计算值，排序
ycalsort=np.zeros(shape=(trainnum*classnum,3))
temp=ycal[0:trainnum,:]
ycalsort[0:trainnum,:]=temp[temp[:,2].argsort()]
temp=ycal[trainnum:2*trainnum,:]
ycalsort[trainnum:2*trainnum,:]=temp[temp[:,2].argsort()]
temp=ycal[2*trainnum:3*trainnum,:]
ycalsort[2*trainnum:3*trainnum,:]=temp[temp[:,2].argsort()]
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.scatter(np.arange(0,trainnum*classnum)/(trainnum*classnum),ycalsort[:,0],marker="o",c="red")
ax1.scatter(np.arange(0,trainnum*classnum)/(trainnum*classnum),ycalsort[:,1],marker="o",c="black")
ax1.scatter(np.arange(0,trainnum*classnum)/(trainnum*classnum),ycalsort[:,2],marker="o",c="blue")
plt.show()