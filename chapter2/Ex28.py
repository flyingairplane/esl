##Ex 2.8
##
##Date:2020/06/27
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#训练集
train2=pd.read_csv("train.2.txt",header=None)
train3=pd.read_csv("train.3.txt",header=None)
xtrainorigin=np.vstack((train2,train3))#两个矩阵按列合并，np.vstack的参数为tuple
x1trainnum=len(train2)
x2trainnum=len(train3)
trainnum=x1trainnum+x2trainnum#训练集大小
ytrain=np.vstack((np.ones(shape=(x1trainnum,1)),np.zeros(shape=(x2trainnum,1))))
# #查看单幅图片
# fig=plt.figure()
# ax1=fig.add_subplot()
# ax1.imshow(np.asmatrix(train2.iloc[0]).reshape(16,16))
# plt.show()

#测试集
testall=pd.read_table("zip.test",index_col=0,sep=' ')
test23=testall.loc[[2,3]]
xtestorigin=np.asarray(test23)
xtest=np.hstack((np.ones(shape=(len(xtestorigin),1)),xtestorigin))#加一列全1
ytest=np.asmatrix(test23.index.values).T
ytest[ytest==2]=1
ytest[ytest==3]=0
testnum=len(ytest)#测试集大小

#线性回归
xtrain=np.hstack((np.ones(shape=(len(xtrainorigin),1)),xtrainorigin))#加一列全1，求beta0
betalinear=np.linalg.inv(xtrain.T.dot(xtrain)).dot(xtrain.T).dot(ytrain)
ylinear=xtrain.dot(betalinear)
yhatlinear=np.where(ylinear>0.5,1,0)
errorlinear=np.count_nonzero(yhatlinear!=ytrain)/(len(ytrain))
print('线性回归训练误差:'+str(errorlinear))

ylineartest=xtest.dot(betalinear)
yhatlineartest=np.where(ylineartest>0.5,1,0)
errorlineartest=np.count_nonzero(yhatlineartest!=ytest)/(len(ytest))
print("线性回归测试误差:"+str(errorlineartest))

#kNN
knum=[1,3,5,7,15]
distancetrain=np.zeros(shape=(trainnum,trainnum))
for i in range(len(xtrainorigin)):
    for j in range(i+1,len(xtrainorigin)):
        distancetrain[i,j]=np.sqrt(sum((xtrainorigin[i,:]-xtrainorigin[j,:])**2))
        distancetrain[j,i]=distancetrain[i,j]
distancetest=np.zeros(shape=(testnum,trainnum))
for i in range(len(xtestorigin)):
    for j in range(0,trainnum):
        distancetest[i,j]=np.sqrt(sum((xtestorigin[i,:]-xtrainorigin[j,:])**2))
classifytrain=np.zeros(shape=(trainnum,1))#训练样本集中每个样本的分类结果
classifytest=np.zeros(shape=(testnum,1))#测试样本集中每个样本的分类结果
errorknntrain=np.zeros(shape=(len(knum),1))
errorknntest=np.zeros(shape=(len(knum),1))
for k in range(len(knum)):
    #训练误差
    for i in range(len(xtrainorigin)):
        classifytrain[i]=1 if np.mean(ytrain[np.argsort(distancetrain[i,:])[0:knum[k]]])>0.5 else 0
    errorknntrain[k]=np.count_nonzero(classifytrain!=ytrain)/(len(ytrain))
    #测试误差
    for i in range(len(xtestorigin)):
        classifytest[i]=1 if np.mean(ytrain[np.argsort(distancetest[i,:])[0:knum[k]]])>0.5 else 0
    errorknntest[k]=np.count_nonzero(classifytest!=ytest)/(len(ytest))

fig1=plt.figure()
ax2=fig1.add_subplot(1,1,1)
ax2.plot(knum,errorknntrain,'bo-',label='knn train')
ax2.plot(knum,errorknntest,'ro-',label='knn test')
plt.legend(loc='upper right')
plt.show()

#线性回归训练误差:0.005759539236861051
#线性回归测试误差:0.04120879120879121